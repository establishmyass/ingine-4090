# ingine for Inkling-2-Mini (this repository's native/ engine): ingine-serve, ingine-bench, ingine-validate.
#
#   nix-build -E 'with import <nixpkgs> { config.allowUnfree = true; }; callPackage ./nix/package.nix { }'
#
# ingineSrc / ingineVersion: the source (default: this checkout) and its version label (not `src` / `version`:
# callPackage would fill those from nixpkgs). cudaArch: 86 (RTX 3090, the default here) or "120a" (RTX 5090). The model is not part of the package: point
# ingine-serve --model-dir at a download of satellitedown/Inkling-2-Mini-abliterated-ingine (see README.md).
{
  lib,
  cudaPackages,
  cmake,
  ninja,
  autoAddDriverRunpath,
  ingineSrc ? lib.cleanSourceWith {
    src = ../.;
    filter =
      path: type:
      let
        rel = lib.removePrefix (toString ../. + "/") (toString path);
      in
      !(lib.hasPrefix "models" rel || lib.hasPrefix ".cuda-toolkit" rel || lib.hasPrefix "native/build" rel
        || lib.hasPrefix ".tools" rel || lib.hasPrefix "work" rel || lib.hasPrefix "result" rel);
  },
  ingineVersion ? "unstable",
  cudaArch ? "86",
}:

cudaPackages.backendStdenv.mkDerivation {
  pname = "inkling-ingine";
  version = ingineVersion;
  src = ingineSrc;

  cmakeDir = "../native";

  nativeBuildInputs = [
    cmake
    ninja
    cudaPackages.cuda_nvcc
    autoAddDriverRunpath
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.cuda_nvtx
    cudaPackages.libcublas
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "CMAKE_CUDA_ARCHITECTURES" cudaArch)
    (lib.cmakeBool "INGINE_BUILD_TESTS" false)
  ];

  # The CMake project has no install rules: the three executables.
  installPhase = ''
    runHook preInstall
    install -Dm755 apps/serve/ingine-serve $out/bin/ingine-serve
    install -Dm755 apps/bench/ingine-bench $out/bin/ingine-bench
    install -Dm755 apps/validate/ingine-validate $out/bin/ingine-validate
    runHook postInstall
  '';

  meta = {
    description = "ingine: CUDA inference engine for Inkling-2-Mini (abliterated), RTX 3090 port with expert offloading and CPU image input";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
    mainProgram = "ingine-serve";
  };
}
