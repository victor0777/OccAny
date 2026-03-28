"""Generate accident cause analysis visualization."""
import json, numpy as np, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
acc = []
for d in sorted(glob.glob('results/accident_analysis/*/analysis.json')):
    with open(d) as f:
        acc.append(json.load(f))
with open('results/non_accident/batch_results.json') as f:
    non_acc = json.load(f)['results']
with open('/home/ktl/projects/accident_analysis/cause_classification_results.json') as f:
    causes = json.load(f)
with open('/home/ktl/projects/accident_analysis/collision_analysis_results.json') as f:
    coll = json.load(f)

for a in acc:
    s = a['stem']
    if s in causes: a['cause'] = causes[s].get('cause', '?')
    if s in coll: a['subject'] = coll[s].get('verified', {}).get('accident_subject', '?')

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. Box plot by cause
ax = axes[0, 0]
cause_order = ['side_collision_intersection', 'unsafe_lane_change_ego', 'solo_collision',
               'rear_end_ego_at_fault', 'sudden_stop_front', 'cut_in_other',
               'observed_accident', 'rear_end_other_at_fault']
cause_data = []
cause_labels = []
for c in cause_order:
    sigs = [a['ego_signal_strength'] for a in acc if a.get('cause') == c]
    if sigs:
        cause_data.append(sigs)
        cause_labels.append(f'{c[:22]}\n(n={len(sigs)})')
cause_data.append([r['ego_signal_strength'] for r in non_acc])
cause_labels.append(f'non_accident\n(n={len(non_acc)})')

bp = ax.boxplot(cause_data, patch_artist=True, vert=True)
colors = ['#e74c3c']*4 + ['#f39c12']*3 + ['#3498db'] + ['#2ecc71']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(cause_labels, fontsize=6, rotation=45, ha='right')
ax.set_ylabel('ego_signal_strength')
ax.set_title('1. ego_signal by Accident Cause\n(red=ego_active, orange=ego_passive, blue=observed, green=non)')

# 2. 4-group comparison
ax = axes[0, 1]
active_causes = ['rear_end_ego_at_fault', 'unsafe_lane_change_ego', 'solo_collision', 'side_collision_intersection']
passive_causes = ['rear_end_other_at_fault', 'cut_in_other', 'sudden_stop_front']
ego_active = [a['ego_signal_strength'] for a in acc if a.get('cause') in active_causes]
ego_passive = [a['ego_signal_strength'] for a in acc if a.get('cause') in passive_causes]
observed = [a['ego_signal_strength'] for a in acc if a.get('cause') == 'observed_accident']
non_sig = [r['ego_signal_strength'] for r in non_acc]

bp2 = ax.boxplot([ego_active, ego_passive, observed, non_sig], patch_artist=True)
for patch, color in zip(bp2['boxes'], ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']):
    patch.set_facecolor(color)
ax.set_xticklabels([f'ego_active\n(n={len(ego_active)})', f'ego_passive\n(n={len(ego_passive)})',
                     f'observed\n(n={len(observed)})', f'non_accident\n(n={len(non_sig)})'], fontsize=9)
ax.set_ylabel('ego_signal_strength')
ax.set_title('2. 4-Group Comparison\n(p=0.043* active-passive, p=0.047* active-observed)')

# 3. signal vs density scatter
ax = axes[1, 0]
for a in acc:
    c = a.get('cause', '?')
    if c in active_causes: color, marker = '#e74c3c', 'o'
    elif c in passive_causes: color, marker = '#f39c12', 's'
    elif c == 'observed_accident': color, marker = '#3498db', '^'
    else: color, marker = 'gray', 'x'
    ax.scatter(a['ego_signal_strength'], a.get('mean_density', 0), c=color, marker=marker, s=30, alpha=0.6)
for r in non_acc:
    ax.scatter(r['ego_signal_strength'], r.get('mean_density', 0), c='#2ecc71', marker='D', s=30, alpha=0.6)
ax.set_xlabel('ego_signal_strength')
ax.set_ylabel('mean_density')
ax.set_title('3. Signal vs Density\n(red=ego_active, orange=passive, blue=observed, green=non)')

# 4. Summary
ax = axes[1, 1]
ax.axis('off')
lines = [
    "OccAny Accident Analysis - Final Results (147 videos)",
    "=" * 55,
    "",
    "Data: 121 daytime accidents + 26 non-accidents",
    "      8 cause types from cause_classification",
    "",
    "Key Findings:",
    "1. ego_active (n=56) vs observed (n=50):",
    "   signal 0.196 vs 0.141, p=0.047*",
    "",
    "2. ego_active vs non_accident (n=26):",
    "   signal 0.196 vs 0.151, p=0.155 ns",
    "",
    "3. ego_active vs ego_passive (n=15):",
    "   signal 0.196 vs 0.127, p=0.043*",
    "",
    "4. Best single-feature F1:",
    "   ego vs observed:   0.642 (th=0.03)",
    "   ego_active vs rest: 0.551 (th=0.05)",
    "",
    "5. risk_events (close_approach) NO correlation",
    "   with ego_signal (r=0.045, p=0.622)",
    "",
    "Conclusion:",
    "ego_signal is ONE useful feature.",
    "Phase 3 multi-feature approach MANDATORY.",
]
ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.suptitle('OccAny Accident Cause Analysis (147 videos, 8 cause types)', fontsize=14)
plt.tight_layout()
plt.savefig('results/accident_cause_analysis.png', dpi=150, bbox_inches='tight')
print('Saved: results/accident_cause_analysis.png')
