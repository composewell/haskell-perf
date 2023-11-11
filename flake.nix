{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    haskell-flake.url = "github:srid/haskell-flake";
    streamly = {
      url =
        "github:composewell/streamly/0ed37ff344c122288e8b4865458908398e45789b";
      flake = false;
    };
  };
  outputs = inputs@{ self, nixpkgs, flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = nixpkgs.lib.systems.flakeExposed;
      imports = [ inputs.haskell-flake.flakeModule ];

      perSystem = { self', pkgs, ... }: {
        haskellProjects.default = {
          packages = {
            streamly-core.source = "${inputs.streamly}/core";
          };
          devShell = { hlsCheck.enable = false; };
        };
      };
    };
}
