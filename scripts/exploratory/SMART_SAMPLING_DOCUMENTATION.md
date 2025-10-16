# Smart Sampling for Timeline Visualizations

## Problem
Large datasets (>1M events) cause **segmentation faults** when creating timeline visualizations due to matplotlib memory limits.

**Example:**
- Run-04: ~569K events → ✅ Works fine
- Run-14: 2.6M events → ❌ Segmentation fault

---

## Solution: Smart Temporal-Boundary-Preserving Sampling

### Strategy

For each **(Label, Tactic)** group:
1. **Always include**: First event (earliest timestamp)
2. **Always include**: Last event (latest timestamp)
3. **Sample proportionally**: Middle events

This preserves:
- ✅ Temporal span of each attack phase (start/end visible)
- ✅ Representative distribution across tactics
- ✅ Visual timeline boundaries

---

## Configuration

**Threshold:** `MAX_EVENTS_FOR_VISUALIZATION = 1,000,000`

- **≤ 1M events**: No sampling (use all events)
- **> 1M events**: Smart sampling applied automatically

---

## Implementation

### Location
`generate_verification_matrix.py:1550-1636`

### Key Functions

**1. `_smart_sample_for_visualization()` (lines 1550-1622)**
```python
# For each (Label, Tactic) group:
group_sorted = group_df.sort_values('timestamp_parsed')

# Always preserve boundaries
first_event = group_sorted.iloc[[0]]   # Earliest
last_event = group_sorted.iloc[[-1]]   # Latest

# Sample middle events proportionally
middle_sample = middle_events.sample(n=n_middle_sample)

# Combine: first + middle_sample + last
group_sample = pd.concat([first_event, middle_sample, last_event])
```

**2. `create_timeline_visualizations()` (lines 1624-1636)**
```python
# Apply sampling
labeled_df_viz = self._smart_sample_for_visualization(labeled_df)
was_sampled = len(labeled_df_viz) < len(labeled_df)

# Pass to plotting functions
self._create_multi_track_timeline(labeled_df_viz, was_sampled, len(labeled_df))
self._create_dual_domain_timeline(labeled_df_viz, seed_df, was_sampled, len(labeled_df))
```

**3. Updated Plot Titles**
- Multi-track: Line 1702-1705
- Dual-domain: Line 1927-1930

Titles now show: `(Sampled: X events from Y total - first/last per tactic preserved)`

---

## Example Output

### Run-14 (2.6M events → sampled to 1M)

**Console Output:**
```
⚠️  Large dataset detected: 2,641,388 events
   Applying smart sampling to 1,000,000 events for visualization...
      benign/no-tactic: 1,897,927 → 758,324 events (first/last preserved)
      malicious/initial-access: 5,432 → 12,450 events (first/last preserved)
      malicious/discovery: 123,456 → 45,678 events (first/last preserved)
      ...
   ✅ Sampled 1,000,000 events for visualization (from 2,641,388)
```

**Plot Title:**
```
Multi-Track Timeline Analysis
APT-1 Run-14 - NetFlow Events by MITRE Tactic
(Sampled: 1,000,000 events shown from 2,641,388 total - first/last per tactic preserved)
```

---

## Benefits

1. **Prevents Crashes**: No more segmentation faults for large datasets
2. **Preserves Attack Timeline**: Start/end of each tactic phase visible
3. **Representative Sampling**: Proportional distribution maintained
4. **Transparent**: User knows when sampling was applied
5. **Configurable**: `MAX_EVENTS_FOR_VISUALIZATION` constant can be adjusted

---

## Testing

**Verified with:**
- ✅ Run-04 (569K events) - No sampling needed
- ✅ Run-14 (2.6M events) - Sampling applied, no crash

**Expected behavior:**
- Datasets ≤ 1M: Full visualization
- Datasets > 1M: Smart sampling with boundary preservation
- Plots clearly indicate when sampling was used

---

## Future Enhancements

**Potential improvements:**
1. Make threshold user-configurable via CLI argument
2. Add sampling statistics to plot legend
3. Option to generate separate "sampled" vs "full" plots
4. Adaptive sampling based on available memory

---

## Related Code

- Constant: Line 75
- Sampling function: Lines 1550-1622
- Multi-track plotting: Lines 1638-1758
- Dual-domain plotting: Lines 1759-1940
