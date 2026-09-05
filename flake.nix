{
  description = "Development environment for chucky";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
    claude-code.inputs.nixpkgs.follows = "nixpkgs";
    claude-code.inputs.flake-utils.follows = "flake-utils";
    codex-cli.url = "github:sadjow/codex-cli-nix";
    codex-cli.inputs.nixpkgs.follows = "nixpkgs";
    codex-cli.inputs.flake-utils.follows = "flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self, # required even if the lsp complains
      nixpkgs,
      flake-utils,
      claude-code,
      codex-cli,
      git-hooks,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            clang-format = {
              enable = true;
              types_or = pkgs.lib.mkForce [
                "c"
                "c++"
                "cuda"
              ];
            };
            gersemi = {
              enable = true;
              name = "gersemi";
              entry = "${pkgs.gersemi}/bin/gersemi -i";
              files = "(\\.cmake$|CMakeLists\\.txt$)";
              pass_filenames = true;
            };
          };
        };

        # Shared dependency lists used by both devShell and the package builds.
        lz4Static = pkgs.lz4.overrideAttrs (old: {
          cmakeFlags = (old.cmakeFlags or [ ]) ++ [ "-DBUILD_STATIC_LIBS=ON" ];
        });
        zstdStatic = pkgs.zstd.override { enableStatic = true; };
        nvcomp53 = pkgs.cudaPackages_13_2.nvcomp.overrideAttrs (_: {
          version = "5.3.0.16";
          src = pkgs.fetchurl {
            url = "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/linux-x86_64/nvcomp-linux-x86_64-5.3.0.16_cuda13-archive.tar.xz";
            hash = "sha256-LDb1r2PDfkr+E9FPkS6EEw5qBfB7BmVHs+AoxMpUyGY=";
          };
        });

        commonBuildInputs = with pkgs; [
          c-blosc
          lz4Static
          zstdStatic
          zlib
          snappy
          # s3 writer
          aws-c-common
          aws-c-cal
          aws-c-io
          aws-c-http
          aws-c-auth
          aws-c-s3
          aws-c-compression
          aws-c-sdkutils
          aws-checksums
          s2n-tls
        ];
        # GPU build inputs are parameterized by CUDA version. The devshell and
        # the default `chucky` package use CUDA 13.2 (Blackwell-era). A
        # CUDA 12.9 variant (`chucky-cuda12`) is also emitted for downstream
        # consumers that pin the older toolkit.
        mkGpuBuildInputs =
          cudaPackages: nvcomp: with pkgs; [
            cudaPackages.cudatoolkit
            nvcomp
            nvcomp.static
            llvmPackages.openmp
          ];

        commonNativeBuildInputs = with pkgs; [
          cmake
          ninja
          pkg-config
        ];

        mkChucky =
          {
            variant, # "gpu" | "cpu"
            cudaPackages ? pkgs.cudaPackages_13_2,
            pname ? null,
          }:
          let
            isGpu = variant == "gpu";
            resolvedPname =
              if pname != null then
                pname
              else if isGpu then
                "chucky"
              else
                "chucky-cpu";
          in
          pkgs.stdenv.mkDerivation {
            pname = resolvedPname;
            version = "0.1.0";
            src = self;

            nativeBuildInputs = commonNativeBuildInputs;
            buildInputs =
              commonBuildInputs ++ pkgs.lib.optionals isGpu (mkGpuBuildInputs cudaPackages cudaPackages.nvcomp);

            cmakeFlags = [
              "-DCHUCKY_ENABLE_GPU=${if isGpu then "ON" else "OFF"}"
              "-DCHUCKY_BUILD_SHARED=ON"
              "-DBUILD_TESTING=OFF"
            ];

            # Tests are disabled in the package builds; CI exercises them via the
            # devShell. CMake still adds `enable_testing()` and the tests subdir,
            # so we override CTest to no-op rather than touch the in-tree CMake.
            doCheck = false;
          };
      in
      {
        checks = {
          inherit pre-commit-check;
        };

        formatter = pkgs.nixfmt-tree;

        packages = {
          # Default GPU package is CUDA 13.2.
          chucky = mkChucky { variant = "gpu"; };
          # Explicit per-CUDA aliases for downstream consumers.
          chucky-cuda13 = mkChucky {
            variant = "gpu";
            cudaPackages = pkgs.cudaPackages_13_2;
            pname = "chucky-cuda13";
          };
          chucky-cuda12 = mkChucky {
            variant = "gpu";
            cudaPackages = pkgs.cudaPackages_12;
            pname = "chucky-cuda12";
          };
          chucky-cpu = mkChucky { variant = "cpu"; };
          default = mkChucky { variant = "gpu"; };
        };

        devShells.default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
          name = "chucky";
          inherit (pre-commit-check) shellHook;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.zlib
          ];

          nativeBuildInputs =
            commonNativeBuildInputs
            ++ (with pkgs; [
              claude-code.packages.${system}.default
              codex-cli.packages.${system}.default
              docker
              gdb
              gh
              man-pages
              man-pages-posix
              neocmakelsp
              nixd
              llvmPackages.llvm # llvm-profdata, llvm-cov for coverage
              cudaPackages_13_2.nsight_systems
              perf
              tmux
              tokei
              awscli2
              nodejs_22
              python3
              uv
            ]);

          buildInputs = commonBuildInputs ++ (mkGpuBuildInputs pkgs.cudaPackages_13_2 nvcomp53);
        };
      }
    );
}
