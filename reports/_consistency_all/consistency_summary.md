# Render Consistency Summary

- case_id: 3D_slicer_files
- analysis_count: 16
- output_dir: /home/rjm/projects/EBUS_simulator/reports/_consistency_all

## Most Divergent Presets

| Preset | Approach | Score | Reasons |
|---|---|---|---|
| station_7_node_a | lms | 0.911 | localizer/physics occupancy delta 0.964; localizer/physics brightness delta 0.338; sector mostly empty (0.974) |
| station_4l_node_a | default | 0.832 | localizer/physics occupancy delta 0.849; localizer/physics brightness delta 0.357; sector mostly empty (0.849); physics normalization upper tail ratio 1.242 |
| station_11rs_node_b | default | 0.779 | localizer/physics occupancy delta 0.748; localizer/physics brightness delta 0.304; sector mostly empty (0.837); physics normalization upper tail ratio 1.339 |
| station_11rs_node_a | default | 0.710 | localizer/physics occupancy delta 0.747; localizer/physics brightness delta 0.301; sector mostly empty (0.837); physics normalization upper tail ratio 1.272 |
| station_11ri_node_a | default | 0.695 | localizer/physics occupancy delta 0.684; localizer/physics brightness delta 0.280; near-field wall occupancy high (0.212); sector mostly empty (0.881) |

## Representative Cases

### Wall Dominant

- `station_11ri_node_a` / `default`: localizer/physics occupancy delta 0.684; localizer/physics brightness delta 0.280; near-field wall occupancy high (0.212); sector mostly empty (0.881)

### Target Prominent

- `station_2l_node_a` / `default`: localizer/physics occupancy delta 0.647; localizer/physics brightness delta 0.278; sector mostly empty (0.793)

### Sparse Dark

- `station_7_node_a` / `lms`: localizer/physics occupancy delta 0.964; localizer/physics brightness delta 0.338; sector mostly empty (0.974)

## Heuristic Breakdown

- target_prominence_disagreements: 1
- occupancy_disagreements: 16
- brightness_disagreements: 16
- edge_target_cases: 0
- wall_dominant_cases: 1
- sparse_sector_cases: 7
- normalization_tail_cases: 3
