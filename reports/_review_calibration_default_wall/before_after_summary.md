# Review Bundle Comparison

- before_summary_json: /Users/russellmiller/Projects/EBUS_simulator/reports/_review_calibration_all/review_summary.json
- after_summary_json: /Users/russellmiller/Projects/EBUS_simulator/reports/_review_calibration_default_wall/review_summary.json
- case_id_match: yes
- matched_entry_count: 16
- before_flagged_count: 8
- after_flagged_count: 8
- resolved_flagged_count: 0
- regressed_flagged_count: 0
- unchanged_flagged_count: 8
- unchanged_clear_count: 8

## Flag Transitions

### Resolved Flags

- none

### New Flags

- none

### Still Flagged

- `station_11l_node_a` / `default`: before=nUS delta 19.67 deg > 10.0 deg; contact delta 3.39 mm > 1.5 mm; after=nUS delta 19.67 deg > 10.0 deg; contact delta 3.39 mm > 1.5 mm
- `station_11ri_node_a` / `default`: before=contact delta 2.56 mm > 1.5 mm; contact refinement remained ambiguous; target contrast -0.014 < 0.000; after=contact delta 2.56 mm > 1.5 mm; contact refinement remained ambiguous; target contrast -0.014 < 0.000
- `station_11rs_node_a` / `default`: before=target region missing from physics eval summary; after=target region missing from physics eval summary
- `station_11rs_node_b` / `default`: before=contact delta 2.31 mm > 1.5 mm; contact refinement remained ambiguous; target contrast -0.009 < 0.000; after=contact delta 2.31 mm > 1.5 mm; contact refinement remained ambiguous; target contrast -0.009 < 0.000
- `station_4l_node_a` / `default`: before=target contrast -0.001 < 0.000; after=target contrast -0.001 < 0.000
- `station_4l_node_b` / `default`: before=target contrast -0.006 < 0.000; after=target contrast -0.006 < 0.000
- `station_4r_node_b` / `default`: before=contact refinement remained ambiguous; after=contact refinement remained ambiguous
- `station_7_node_a` / `lms`: before=target contrast -0.001 < 0.000; vessel contrast -0.003 > -0.010; after=target contrast -0.001 < 0.000; vessel contrast -0.003 > -0.010

## Contrast Table

| Preset | Approach | Transition | Before Target | After Target | Before Wall | After Wall | Before Vessel | After Vessel |
|---|---|---|---|---|---|---|---|---|
| station_10r_node_a | default | still_clear | 0.013 | 0.013 | 0.848 | 0.848 | -0.049 | -0.049 |
| station_10r_node_b | default | still_clear | 0.040 | 0.040 | 0.948 | 0.948 | -0.037 | -0.037 |
| station_11l_node_a | default | still_flagged | 0.001 | 0.001 | 0.165 | 0.165 | -0.035 | -0.035 |
| station_11ri_node_a | default | still_flagged | -0.014 | -0.014 | 0.473 | 0.473 | -0.014 | -0.014 |
| station_11rs_node_a | default | still_flagged | n/a | n/a | 0.508 | 0.508 | -0.029 | -0.029 |
| station_11rs_node_b | default | still_flagged | -0.009 | -0.009 | 0.845 | 0.845 | -0.026 | -0.026 |
| station_2l_node_a | default | still_clear | 0.070 | 0.070 | 0.931 | 0.931 | -0.024 | -0.024 |
| station_2r_node_a | default | still_clear | 0.041 | 0.041 | 0.829 | 0.829 | n/a | n/a |
| station_4l_node_a | default | still_flagged | -0.001 | -0.001 | 0.852 | 0.852 | -0.031 | -0.031 |
| station_4l_node_b | default | still_flagged | -0.006 | -0.006 | 0.949 | 0.949 | -0.045 | -0.045 |
| station_4l_node_c | default | still_clear | 0.021 | 0.021 | 0.863 | 0.863 | -0.044 | -0.044 |
| station_4r_node_a | default | still_clear | 0.020 | 0.020 | 0.884 | 0.884 | -0.050 | -0.050 |
| station_4r_node_b | default | still_flagged | 0.038 | 0.038 | 0.874 | 0.874 | -0.047 | -0.047 |
| station_4r_node_c | default | still_clear | 0.057 | 0.057 | 0.965 | 0.965 | -0.025 | -0.025 |
| station_7_node_a | lms | still_flagged | -0.001 | -0.001 | 0.039 | 0.039 | -0.003 | -0.003 |
| station_7_node_a | rms | still_clear | 0.012 | 0.012 | 0.178 | 0.178 | -0.045 | -0.045 |
