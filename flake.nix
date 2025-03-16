{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in with pkgs; {
        devShells.default = mkShell {

          packages = [
            uv

            # cmake
            # pkg-config
            # sentencepiece
          ];

          buildInputs = lib.optionals stdenv.isDarwin [
            # darwin.apple_sdk.frameworks.CoreFoundation
            # darwin.apple_sdk.frameworks.Security
            # darwin.libobjc
            # darwin.apple_sdk.frameworks.CoreServices
            # darwin.apple_sdk.frameworks.System
          ];

        };
      }
    );
}
