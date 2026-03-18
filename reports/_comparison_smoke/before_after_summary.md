# Review Bundle Comparison

- before_summary_json: /Users/russellmiller/Projects/EBUS_simulator/reports/preset_review_20260316/review_summary.json
- after_summary_json: /Users/russellmiller/Projects/EBUS_simulator/reports/preset_review_stabilized/review_summary.json
- case_id_match: yes
- matched_entry_count: 16
- before_flagged_count: 8
- after_flagged_count: 4
- resolved_flagged_count: 4
- regressed_flagged_count: 0
- unchanged_flagged_count: 4
- unchanged_clear_count: 8

## Flag Transitions

### Resolved Flags

- `station_11rs_node_a` / `default`: before=nUS delta 13.15 deg > 10.0 deg; target not in displayed fan sector; after=none
- `station_2r_node_a` / `default`: before=nUS delta 20.23 deg > 10.0 deg; after=none
- `station_4l_node_a` / `default`: before=nUS delta 16.38 deg > 10.0 deg; after=none
- `station_7_node_a` / `lms`: before=nUS delta 15.00 deg > 10.0 deg; target not in displayed fan sector; after=none

### New Flags

- none

### Still Flagged

- `station_11l_node_a` / `default`: before=nUS delta 44.72 deg > 10.0 deg; contact delta 2.38 mm > 1.5 mm; station overlap 0.0000 < 0.0030; after=nUS delta 19.67 deg > 10.0 deg; contact delta 3.39 mm > 1.5 mm
- `station_11ri_node_a` / `default`: before=nUS delta 13.10 deg > 10.0 deg; contact delta 2.11 mm > 1.5 mm; contact refinement remained ambiguous; after=contact delta 2.56 mm > 1.5 mm; contact refinement remained ambiguous
- `station_11rs_node_b` / `default`: before=contact delta 1.76 mm > 1.5 mm; contact refinement remained ambiguous; after=contact delta 2.31 mm > 1.5 mm; contact refinement remained ambiguous
- `station_4r_node_b` / `default`: before=contact refinement remained ambiguous; after=contact refinement remained ambiguous

## Contrast Table

| Preset | Approach | Transition | Before Target | After Target | Before Wall | After Wall | Before Vessel | After Vessel |
|---|---|---|---|---|---|---|---|---|
| station_10r_node_a | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_10r_node_b | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_11l_node_a | default | still_flagged | n/a | n/a | n/a | n/a | n/a | n/a |
| station_11ri_node_a | default | still_flagged | n/a | n/a | n/a | n/a | n/a | n/a |
| station_11rs_node_a | default | resolved | n/a | n/a | n/a | n/a | n/a | n/a |
| station_11rs_node_b | default | still_flagged | n/a | n/a | n/a | n/a | n/a | n/a |
| station_2l_node_a | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_2r_node_a | default | resolved | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4l_node_a | default | resolved | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4l_node_b | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4l_node_c | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4r_node_a | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4r_node_b | default | still_flagged | n/a | n/a | n/a | n/a | n/a | n/a |
| station_4r_node_c | default | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
| station_7_node_a | lms | resolved | n/a | n/a | n/a | n/a | n/a | n/a |
| station_7_node_a | rms | still_clear | n/a | n/a | n/a | n/a | n/a | n/a |
