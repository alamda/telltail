#!/usr/bin/env python3
"""Create derived LANL auth subsets from a single pass over auth.txt.

The source file is large, so each input row is written to every matching output
during the same read. Outputs are gzipped CSV files under data/lanl/derived/.

  smoke              first N auth rows
  redteam_auth       rows matching an event in redteam.txt
  compromised_users  full auth history for users named in redteam.txt
  benign_sample      deterministic 1/N sample of all other rows
  failures           rows with failed authentication outcomes

auth.txt:    time, src_user, dst_user, src_comp, dst_comp, auth_type,
             logon_type, orient, outcome
redteam.txt: time, user, src_comp, dst_comp
"""
import argparse
import gzip
import sys
import time
import zlib
from pathlib import Path

HEADER = ("time,src_user,dst_user,src_comp,dst_comp,"
          "auth_type,logon_type,auth_orientation,outcome,label\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auth", type=Path, default=Path("data/lanl/auth.txt"))
    parser.add_argument("--redteam", type=Path, default=Path("data/lanl/redteam.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/lanl/derived"))
    parser.add_argument("--smoke-rows", type=int, default=200_000)
    parser.add_argument("--sample-denom", type=int, default=1000,
                        help="benign sample keeps 1/N of the other accounts")
    parser.add_argument("--total-rows", type=int, default=1_051_430_459,
                        help="for the ETA readout only")
    parser.add_argument("--progress-every", type=int, default=10_000_000)
    args = parser.parse_args()
    # TODO: reject --sample-denom <= 0.
    # TODO: reject --progress-every <= 0.
    return args


def load_redteam(path):
    """Read redteam.txt -> (set of (timestamp, user, src_comp, dst_comp), users)."""
    events, users = set(), set()
    with open(path) as redteam_file:
        for line in redteam_file:
            line = line.strip()
            if line:
                timestamp, user, src_comp, dst_comp = line.split(",")
                events.add((timestamp, user, src_comp, dst_comp))
                users.add(user)
    return events, users


def open_writers(out_dir):
    def writer(name):
        out_file = gzip.open(out_dir / name, "wt", compresslevel=4)
        out_file.write(HEADER)
        return out_file

    return (
        writer("smoke.csv.gz"),
        writer("redteam_auth.csv.gz"),
        writer("compromised_users.csv.gz"),
        writer("benign_sample.csv.gz"),
        writer("failures.csv.gz"),
    )


def print_progress(rows_read, started_at, total_rows, redteam_rows_written,
                   compromised_rows_written, benign_rows_written,
                   failure_rows_written):
    million_rows_per_second = rows_read / (time.time() - started_at) / 1e6
    eta_minutes = ((total_rows - rows_read)
                   / (million_rows_per_second * 1e6) / 60)
    print(f"{rows_read/1e6:.0f}M rows | rt={redteam_rows_written} "
          f"comp={compromised_rows_written} benign={benign_rows_written} "
          f"fail={failure_rows_written} | {million_rows_per_second:.2f}M/s, "
          f"ETA {eta_minutes:.0f} min",
          file=sys.stderr)


def main():
    args = parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    events, flagged = load_redteam(args.redteam)
    print(f"redteam.txt: {len(events)} raw events, {len(flagged)} users", file=sys.stderr)

    (smoke_file, redteam_auth_file, compromised_users_file,
     benign_sample_file, failures_file) = open_writers(args.out)
    # TODO: use context managers so gzip outputs close on errors.

    rows_read = 0
    smoke_rows_written = 0
    redteam_rows_written = 0
    compromised_rows_written = 0
    benign_rows_written = 0
    failure_rows_written = 0
    matched_events = set()
    started_at = time.time()

    with open(args.auth, buffering=1 << 20) as auth:
        for line in auth:
            rows_read += 1
            (timestamp, src_user, dst_user, src_comp, dst_comp,
             _auth_type, _logon_type, _orient, outcome) = line.split(",")
            outcome = outcome.strip()

            src_event = (timestamp, src_user, src_comp, dst_comp)
            dst_event = (timestamp, dst_user, src_comp, dst_comp)
            is_redteam_event = src_event in events or dst_event in events
            if src_event in events:
                matched_events.add(src_event)
            if dst_event in events:
                matched_events.add(dst_event)
            row = line.rstrip("\n") + ("," + ("1" if is_redteam_event else "0")) + "\n"

            if rows_read <= args.smoke_rows:
                smoke_file.write(row); smoke_rows_written += 1
            if outcome.startswith("Fail"):
                failures_file.write(row); failure_rows_written += 1
            if is_redteam_event:
                redteam_auth_file.write(row); redteam_rows_written += 1

            if src_user in flagged or dst_user in flagged:
                compromised_users_file.write(row); compromised_rows_written += 1
            else:
                # per-row key includes timestamp. drop it to sample whole edges instead.
                sample_key = timestamp + src_user + dst_user + src_comp + dst_comp
                if zlib.crc32(sample_key.encode()) % args.sample_denom == 0:
                    benign_sample_file.write(row); benign_rows_written += 1

            if rows_read % args.progress_every == 0:
                print_progress(rows_read, started_at, args.total_rows,
                               redteam_rows_written, compromised_rows_written,
                               benign_rows_written, failure_rows_written)

    for out_file in (smoke_file, redteam_auth_file, compromised_users_file,
                     benign_sample_file, failures_file):
        out_file.close()

    print(f"\ndone in {(time.time()-started_at)/60:.1f} min. read {rows_read} rows.",
          file=sys.stderr)
    print(f"  smoke={smoke_rows_written} redteam_auth={redteam_rows_written} "
          f"compromised_users={compromised_rows_written} "
          f"benign_sample={benign_rows_written} failures={failure_rows_written}",
          file=sys.stderr)
    print(f"  matched {len(matched_events)}/{len(events)} unique redteam events",
          file=sys.stderr)
    if len(matched_events) < len(events):
        missing = len(events) - len(matched_events)
        print(f"WARNING: {missing} redteam event(s) not found in auth.txt. "
              f"Known LANL gap: a few redteam events have no exact "
              f"(time, user, src_comp, dst_comp) row in auth.txt, not a match-key bug.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
