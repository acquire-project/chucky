{
  description = "Development environment for chucky";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    claude-code.url = "github:sadjow/claude-code-nix";
    claude-code.inputs.nixpkgs.follows = "nixpkgs";
    claude-code.inputs.flake-utils.follows = "flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      self, # required even if the lsp complains
      nixpkgs,
      flake-utils,
      claude-code,
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

        commonBuildInputs = with pkgs; [
          c-blosc
          lz4Static
          zstdStatic
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
        gpuBuildInputs = with pkgs; [
          cudaPackages.cudatoolkit
          cudaPackages.nvcomp
          cudaPackages.nvcomp.static
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
          }:
          let
            isGpu = variant == "gpu";
            pname = if isGpu then "chucky" else "chucky-cpu";
          in
          pkgs.stdenv.mkDerivation {
            inherit pname;
            version = "0.1.0";
            src = self;

            nativeBuildInputs = commonNativeBuildInputs;
            buildInputs = commonBuildInputs ++ pkgs.lib.optionals isGpu gpuBuildInputs;

            cmakeFlags = [
              "-DCHUCKY_ENABLE_GPU=${if isGpu then "ON" else "OFF"}"
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
          chucky = mkChucky { variant = "gpu"; };
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
              docker
              gdb
              gh
              man-pages
              man-pages-posix
              neocmakelsp
              nixd
              llvmPackages.llvm # llvm-profdata, llvm-cov for coverage
              perf
              tmux
              tokei
              awscli2
              python3
              uv
            ]);

          buildInputs = commonBuildInputs ++ gpuBuildInputs;
        };
      }
    );
}
