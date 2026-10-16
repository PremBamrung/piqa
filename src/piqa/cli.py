"""`piqa` command line."""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

from piqa import __version__


def cmd_ingest(args: argparse.Namespace) -> None:
    from piqa.ingest import ingest

    dest = args.dest
    src = args.source or dest
    copy = src.resolve() != dest.resolve() and not args.move
    ingest(src, dest, copy=copy, dry_run=args.dry_run)


def _embed(args: argparse.Namespace) -> Path:
    from piqa.config import load_dinov3_config
    from piqa.embed import embed

    cfg = load_dinov3_config()
    if args.model:
        cfg.model = args.model
    return embed(
        args.folder,
        cfg,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_images=args.max_images,
        fp32=args.fp32,
    )


def cmd_embed(args: argparse.Namespace) -> None:
    _embed(args)


def cmd_map(args: argparse.Namespace) -> None:
    from piqa.viewers import reduce, write_viewers

    emb_path = _embed(args)
    rels, coords = reduce(
        emb_path,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=None if args.parallel else args.seed,
    )
    map_html, wall_html = write_viewers(args.folder, rels, coords, color_by=args.color_by)
    if args.open:
        webbrowser.open((wall_html if args.open == "wall" else map_html).resolve().as_uri())


def cmd_diff(args: argparse.Namespace) -> None:
    from piqa.diff import diff

    diff(args.a, args.b, verbose=args.verbose)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="piqa", description="Photo culling and exploration.")
    parser.add_argument("--version", action="version", version=f"piqa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="sort camera files into <dest>/<date>/<EXT>/")
    p.add_argument(
        "source",
        nargs="?",
        type=Path,
        help="card or folder to import (default: sort the loose files in DEST)",
    )
    p.add_argument(
        "--dest", type=Path, default=Path("Import"), help="destination root (default: ./Import)"
    )
    p.add_argument(
        "--move",
        action="store_true",
        help="move instead of copy when importing from another folder",
    )
    p.add_argument("-n", "--dry-run", action="store_true", help="show what would happen")
    p.set_defaults(func=cmd_ingest)

    def add_embed_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("folder", type=Path, help="photo folder (searched recursively)")
        p.add_argument("--model", help="DINOv3 hub model, e.g. dinov3_vitb16 (default: piqa.toml)")
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--workers", type=int, help="image decoding processes (default: up to 8)")
        p.add_argument("--max-images", type=int, help="only the first N images (quick tests)")
        p.add_argument("--fp32", action="store_true", help="disable fp16 on CUDA")

    p = sub.add_parser("embed", help="compute DINOv3 embeddings + thumbnails into <folder>/.piqa/")
    add_embed_args(p)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("map", help="embed, project with UMAP, write map.html + wall.html")
    add_embed_args(p)
    p.add_argument("--n-neighbors", type=int, default=15, help="UMAP local/global balance")
    p.add_argument("--min-dist", type=float, default=0.1, help="UMAP cluster tightness")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--parallel",
        action="store_true",
        help="drop the seed so UMAP uses all cores (not reproducible)",
    )
    p.add_argument(
        "--color-by",
        choices=["folder", "top", "none"],
        default="folder",
        help="colour points by parent folder, top-level folder, or not at all",
    )
    p.add_argument(
        "--open",
        nargs="?",
        const="wall",
        choices=["wall", "map"],
        help="open a viewer in the browser when done (default: wall)",
    )
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("diff", help="count files in A missing from B (same relative path)")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("-v", "--verbose", action="store_true", help="list the missing files")
    p.set_defaults(func=cmd_diff)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
