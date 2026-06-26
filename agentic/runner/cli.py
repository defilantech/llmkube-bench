import argparse, glob, json, os, time
from .config import load_config
from .record import RunRecord
from .report import render_markdown
from .cluster import KubectlCluster
from .orchestrate import run_matrix


def _gh_issue_body(repo: str, issue: int) -> str:
    import subprocess
    p = subprocess.run(["gh", "issue", "view", str(issue), "--repo", repo,
                        "--json", "body", "--jq", ".body"], capture_output=True, text=True)
    return p.stdout


def cmd_run(config_path: str, records_dir: str, endpoint: str, token: str,
            harness_version: str):
    cfg = load_config(config_path)
    cluster = KubectlCluster(cfg.hardware.context, cfg.hardware.namespace,
                             cfg.hardware.router, cfg.hardware.gateway_backend_index)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    records = run_matrix(cfg, cluster=cluster, harness_version=harness_version, now=now,
                         prompt_for=_gh_issue_body, endpoint=endpoint, token=token)
    os.makedirs(records_dir, exist_ok=True)
    for r in records:
        fname = f"{r.model}-{r.task_id}.json".replace("/", "_")
        with open(os.path.join(records_dir, fname), "w") as f:
            f.write(r.to_json())
    print(f"wrote {len(records)} records to {records_dir}")


def cmd_report(records_dir: str, out: str):
    records = [RunRecord.from_json(open(p).read())
               for p in sorted(glob.glob(os.path.join(records_dir, "*.json")))]
    md = render_markdown(records)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(md)
    print(f"wrote report to {out} ({len(records)} records)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--records", default="records")
    r.add_argument("--endpoint", default="http://localhost:18080")
    r.add_argument("--token", default="")
    r.add_argument("--harness-version", default="unknown")
    rep = sub.add_parser("report")
    rep.add_argument("--records", default="records")
    rep.add_argument("--out", default="reports/latest.md")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        cmd_run(args.config, args.records, args.endpoint, args.token, args.harness_version)
    else:
        cmd_report(args.records, args.out)


if __name__ == "__main__":
    main()
