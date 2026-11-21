#include "serve/openai.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace ingine::serve {
namespace {

struct Bound {
  double value;
  const char* text;  // Python's str() of the bound, as printed in the error message
};

const Json* get(const Json& body, const char* key) {
  auto it = body.find(key);
  return it == body.end() || it->is_null() ? nullptr : &*it;
}

std::string range_message(const char* key, const std::optional<Bound>& lo, const std::optional<Bound>& hi) {
  return std::string("'") + key + "' must be in [" + (lo ? lo->text : "None") + ", " + (hi ? hi->text : "None") +
         "]";
}

int64_t saturate(double v) {
  if (v >= 9.2233720368547758e18) return std::numeric_limits<int64_t>::max();
  if (v <= -9.2233720368547758e18) return std::numeric_limits<int64_t>::min();
  return static_cast<int64_t>(v);
}

// server._num(body, key, int, lo, hi)
std::optional<int64_t> num_int(const Json& body, const char* key, std::optional<Bound> lo = std::nullopt,
                               std::optional<Bound> hi = std::nullopt) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number() ||
      (v->is_number_float() && std::trunc(v->get<double>()) != v->get<double>())) {
    throw BadRequest(std::string("'") + key + "' must be an integer", key);
  }
  int64_t value;
  if (v->is_number_unsigned()) {
    value = int64_t(std::min<uint64_t>(v->get<uint64_t>(), uint64_t(std::numeric_limits<int64_t>::max())));
  } else if (v->is_number_integer()) {
    value = v->get<int64_t>();
  } else {
    value = saturate(v->get<double>());
  }
  if ((lo && double(value) < lo->value) || (hi && double(value) > hi->value)) {
    throw BadRequest(range_message(key, lo, hi), key);
  }
  return value;
}

// server._num(body, key, float, lo, hi)
std::optional<double> num_float(const Json& body, const char* key, std::optional<Bound> lo, std::optional<Bound> hi) {
  const Json* v = get(body, key);
  if (!v) return std::nullopt;
  if (v->is_boolean() || !v->is_number()) throw BadRequest(std::string("'") + key + "' must be a number", key);
  const double value = v->get<double>();
  if ((lo && value < lo->value) || (hi && value > hi->value)) throw BadRequest(range_message(key, lo, hi), key);
  return value;
}

int clamp_int(int64_t v) { return int(std::clamp<int64_t>(v, INT_MIN, INT_MAX)); }

}  // namespace

bool truthy(const Json& value) {
  switch (value.type()) {
    case Json::value_t::null: return false;
    case Json::value_t::boolean: return value.get<bool>();
    case Json::value_t::number_integer: return value.get<int64_t>() != 0;
    case Json::value_t::number_unsigned: return value.get<uint64_t>() != 0;
    case Json::value_t::number_float: return value.get<double>() != 0.0;
    case Json::value_t::string: return !value.get_ref<const std::string&>().empty();
    case Json::value_t::array:
    case Json::value_t::object: return !value.empty();
    default: return true;
  }
}

namespace {

int b64_value(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+' || c == '-') return 62;
  if (c == '/' || c == '_') return 63;
  return -1;
}

std::vector<uint8_t> base64_decode(std::string_view s) {
  std::vector<uint8_t> out;
  out.reserve(s.size() / 4 * 3);
  uint32_t acc = 0;
  int bits = 0;
  for (char c : s) {
    if (c == '=') break;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    const int v = b64_value(c);
    if (v < 0) throw vision::ImageError("image data is not valid base64");
    acc = (acc << 6) | uint32_t(v);
    bits += 6;
    if (bits >= 8) {
      bits -= 8;
      out.push_back(uint8_t(acc >> bits));
      acc &= (1u << bits) - 1;
    }
  }
  return out;
}

// Encodes the request's images and expands each placeholder token of `ids` into the image's tokens.
std::vector<model::PromptImage> expand_images(const Json& messages, std::vector<int32_t>& ids,
                                              vision::ImageEncoder* vision) {
  const std::vector<std::string> sources = text::image_sources(messages);
  if (sources.empty()) return {};
  if (vision == nullptr)
    throw BadRequest("this server does not accept images (no vision tower loaded)", "messages");
  std::vector<vision::EncodedImage> encoded;
  for (const std::string& src : sources) {
    try {
      encoded.push_back(vision->encode(decode_image_url(src)));
    } catch (const vision::ImageError& e) {
      throw BadRequest(std::string("invalid image: ") + e.what(), "messages");
    } catch (const std::bad_alloc&) {
      throw BadRequest("image encoding failed: out of memory", "messages", 500);
    } catch (const std::runtime_error& e) {
      throw BadRequest(std::string("image encoding failed: ") + e.what(), "messages", 500);
    }
  }
  std::vector<model::PromptImage> images;
  std::vector<int32_t> out;
  std::size_t total = ids.size();
  for (const auto& e : encoded) total += std::size_t(e.tokens()) - 1;
  out.reserve(total);
  for (int32_t id : ids) {
    if (id != vision::kImageToken) {
      out.push_back(id);
      continue;
    }
    if (images.size() == encoded.size()) throw BadRequest("more image placeholders than images", "messages");
    const vision::EncodedImage& e = encoded[images.size()];
    images.push_back({int(out.size()), e.tokens(), e.hash, e.embeds});
    out.insert(out.end(), std::size_t(e.tokens()), vision::kImageToken);
  }
  if (images.size() != encoded.size()) throw BadRequest("image placeholders do not match the images", "messages");
  ids = std::move(out);
  return images;
}

}  // namespace

std::vector<uint8_t> decode_image_url(const std::string& url) {
  std::string_view s(url);
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
  auto starts = [&](std::string_view p) {
    if (s.size() < p.size()) return false;
    for (std::size_t i = 0; i < p.size(); ++i)
      if (std::tolower(static_cast<unsigned char>(s[i])) != p[i]) return false;
    return true;
  };
  if (starts("data:")) {
    const std::size_t comma = s.find(',');
    if (comma == std::string_view::npos) throw vision::ImageError("malformed data: URL");
    if (s.substr(5, comma - 5).find(";base64") == std::string_view::npos)
      throw vision::ImageError("data: URLs must be base64 encoded (data:image/...;base64,...)");
    std::vector<uint8_t> bytes = base64_decode(s.substr(comma + 1));
    if (bytes.empty()) throw vision::ImageError("the data: URL holds no data");
    return bytes;
  }
  if (starts("http://") || starts("https://") || starts("file:") || starts("ftp://"))
    throw vision::ImageError("send images as data: URLs (base64); this server does not fetch URLs or files");
  if (s.empty()) throw vision::ImageError("empty image reference");
  try {
    return base64_decode(s);  // bare base64 (transformers {"type": "image", "image": "<base64>"})
  } catch (const vision::ImageError&) {
    throw vision::ImageError("unsupported image reference: expected a data: URL (base64)");
  }
}

ChatRequest parse_request(const Json& body, const text::ChatFormat& fmt, int max_model_len,
                          const model::Sampling& defaults, vision::ImageEncoder* vision) {
  if (num_int(body, "n", Bound{1, "1"}, Bound{1, "1"}).value_or(1) != 1) {
    throw BadRequest("only n=1 is supported", "n");
  }
  Json kwargs = Json::object();
  if (auto it = body.find("chat_template_kwargs"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) {
      throw BadRequest("'chat_template_kwargs' must be an object", "chat_template_kwargs");
    }
    kwargs = *it;
  }
  // body.get("reasoning_effort", kwargs.pop("reasoning_effort", None))
  Json effort = kwargs.contains("reasoning_effort") ? kwargs["reasoning_effort"] : Json();
  kwargs.erase("reasoning_effort");
  if (auto it = body.find("reasoning_effort"); it != body.end()) effort = *it;
  // kwargs.pop("enable_thinking", body.get("enable_thinking", True))
  Json enable_thinking = true;
  if (auto it = body.find("enable_thinking"); it != body.end()) enable_thinking = *it;
  if (kwargs.contains("enable_thinking")) {
    enable_thinking = kwargs["enable_thinking"];
    kwargs.erase("enable_thinking");
  }
  if (!enable_thinking.is_boolean()) throw BadRequest("'enable_thinking' must be a boolean", "enable_thinking");

  text::ChatOptions options;
  options.reasoning_effort = std::move(effort);
  options.enable_thinking = enable_thinking.get<bool>();
  options.template_kwargs = std::move(kwargs);
  const auto messages = body.find("messages");
  const auto tools = body.find("tools");
  static const Json kAbsent;  // null: no messages/tools key (passed by reference, not a deep copy of the body)
  text::RenderedPrompt prompt = fmt.render(messages == body.end() ? kAbsent : *messages,
                                           tools == body.end() ? kAbsent : *tools, options);
  std::vector<model::PromptImage> images =
      expand_images(messages == body.end() ? kAbsent : *messages, prompt.ids, vision);

  const int64_t room = int64_t(max_model_len) - int64_t(prompt.ids.size());
  if (room <= 0) {
    throw BadRequest("prompt has " + std::to_string(prompt.ids.size()) + " tokens; the model context is " +
                         std::to_string(max_model_len),
                     "messages");
  }
  std::optional<int64_t> max_tokens = num_int(body, "max_completion_tokens", Bound{1, "1"});
  if (!max_tokens) max_tokens = num_int(body, "max_tokens", Bound{1, "1"});

  ChatRequest req;
  req.sampling.temperature =
      float(num_float(body, "temperature", Bound{0.0, "0.0"}, Bound{2.0, "2.0"}).value_or(defaults.temperature));
  req.sampling.top_p = float(num_float(body, "top_p", Bound{0.0, "0.0"}, Bound{1.0, "1.0"}).value_or(defaults.top_p));
  req.sampling.top_k =
      clamp_int(std::max<int64_t>(0, num_int(body, "top_k", Bound{-1, "-1"}).value_or(defaults.top_k)));
  if (auto seed = num_int(body, "seed")) req.sampling.seed = static_cast<uint64_t>(*seed);
  if (req.sampling.top_p <= 0.0f) throw BadRequest("'top_p' must be in (0, 1]", "top_p");

  if (const Json* stop = get(body, "stop")) {
    if (stop->is_string()) {
      req.stop.push_back(stop->get<std::string>());
    } else if (stop->is_array() &&
               std::all_of(stop->begin(), stop->end(), [](const Json& s) { return s.is_string(); })) {
      for (const Json& s : *stop) req.stop.push_back(s.get<std::string>());
    } else {
      throw BadRequest("'stop' must be a string or a list of strings", "stop");
    }
  }
  if (auto it = body.find("stream"); it != body.end()) {
    if (!it->is_boolean()) throw BadRequest("'stream' must be a boolean", "stream");
    req.stream = it->get<bool>();
  }
  if (auto it = body.find("stream_options"); it != body.end() && truthy(*it)) {
    if (!it->is_object()) throw BadRequest("'stream_options' must be an object", "stream_options");
    if (auto u = it->find("include_usage"); u != it->end()) req.include_usage = truthy(*u);
  }
  req.max_new_tokens = int(std::min(max_tokens.value_or(room), room));
  req.prompt_ids = std::move(prompt.ids);
  req.prefill_ids = std::move(prompt.prefill_ids);
  req.images = std::move(images);
  return req;
}

Json error_body(int status, std::string_view message, const std::optional<std::string>& param) {
  const char* kind = status == 400 ? "invalid_request_error"
                     : status == 404 ? "not_found_error"
                     : status == 503 ? "server_busy"
                                     : "server_error";
  return Json{{"error",
               {{"message", message}, {"type", kind}, {"param", param ? Json(*param) : Json()}, {"code", status}}}};
}

Json delta_json(const text::Delta& d) {
  if (!d.reasoning_content.empty()) return Json{{"reasoning_content", d.reasoning_content}};
  if (!d.content.empty()) return Json{{"content", d.content}};
  const text::ToolCallDelta& tc = *d.tool_call;
  if (tc.id) {
    return Json{{"tool_calls",
                 Json::array({Json{{"index", tc.index},
                                   {"id", *tc.id},
                                   {"type", "function"},
                                   {"function", {{"name", tc.name ? Json(*tc.name) : Json()},
                                                 {"arguments", tc.arguments}}}}})}};
  }
  return Json{{"tool_calls", Json::array({Json{{"index", tc.index}, {"function", {{"arguments", tc.arguments}}}}})}};
}

std::string dump(const Json& value) { return value.dump(-1, ' ', false, Json::error_handler_t::replace); }

}  // namespace ingine::serve
