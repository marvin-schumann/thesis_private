# Section 5 Figure Placement Guide

## Recommended Figure Placement in Thesis

### Section 5.4: Simulator Validation

**Insert**: Figure 5.1 (Simulator Validation Results)
**After paragraph discussing**: Validation metrics and acceptance criteria
**Caption reference**: "Figure 5.1 shows the simulator validation results..."

**Insert**: Figure 5.6 (Predicted vs. Actual Costs)
**After paragraph discussing**: Cost prediction correlation analysis
**Caption reference**: "As shown in Figure 5.6, the correlation between predicted and actual costs..."

### Section 5.5: Action Masking Necessity

**Insert**: Figure 5.7 (Action Masking Mechanism)
**After paragraph describing**: How action masking works technically
**Caption reference**: "The action masking mechanism (Figure 5.7) restricts the policy..."

**Insert**: Figure 5.2 (Action Masking Impact)
**After paragraph presenting**: Efficiency comparison results
**Caption reference**: "Figure 5.2 demonstrates the dramatic impact of action masking..."

### Section 5.6: Policy Performance Comparison

**Insert**: Figure 5.3 (Policy Performance Comparison)
**After paragraph introducing**: Overall performance results
**Caption reference**: "Figure 5.3 presents the average cost per call for all evaluated policies..."

**Insert**: Figure 5.5 (Cost Distribution)
**After paragraph discussing**: Performance consistency and variance
**Caption reference**: "The cost distributions (Figure 5.5) reveal differences in consistency..."

### Section 5.7: RL Training Analysis

**Insert**: Figure 5.4 (Learning Curve)
**After paragraph discussing**: Training dynamics and convergence
**Caption reference**: "Figure 5.4 shows the training dynamics over 200,000 timesteps..."

---

## LaTeX/Word Insertion Instructions

### For LaTeX:

```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{figures/figure5_1_simulator_validation.png}
  \caption{[Insert caption from FIGURE_CAPTIONS.txt]}
  \label{fig:simulator_validation}
\end{figure}
```

**Notes**:
- Adjust `width=0.8\textwidth` as needed (typically 0.7-1.0)
- Use `[htbp]` placement specifier for "here, top, bottom, page"
- Create unique `\label{}` for each figure for cross-referencing
- Reference in text with `\ref{fig:simulator_validation}`

### For Microsoft Word:

1. **Insert** > **Picture** > Select PNG file from `figures/` directory
2. Right-click image > **Insert Caption**
3. Copy caption from `FIGURE_CAPTIONS.txt`
4. Set caption numbering format: "Figure 5.X"
5. Ensure "Figure" label is selected in Caption dialog
6. Position: Below image (standard convention)

---

## Figure Order in Thesis

Recommended order of appearance (following narrative flow):

1. **Figure 5.1** - Section 5.4.1 (Simulator Validation Results)
2. **Figure 5.6** - Section 5.4.2 (Scatter plot detail)
3. **Figure 5.7** - Section 5.5.1 (Action masking explanation)
4. **Figure 5.2** - Section 5.5.2 (Action masking impact)
5. **Figure 5.3** - Section 5.6.1 (Overall performance)
6. **Figure 5.5** - Section 5.6.2 (Distribution analysis)
7. **Figure 5.4** - Section 5.7 (Training dynamics)

---

## Cross-Reference Examples

In text, refer to figures as:
- "As shown in Figure 5.1..."
- "Figure 5.3 demonstrates that..."
- "The action masking mechanism (Figure 5.7) restricts..."
- "Training dynamics (Figure 5.4) reveal..."

**LaTeX Cross-References**:
```latex
As shown in Figure~\ref{fig:simulator_validation}...
The results (Figure~\ref{fig:policy_performance}) indicate...
```

**Pro tip**: Use `~` (non-breaking space) between "Figure" and the reference to prevent line breaks.

---

## Figure Sizing Guidelines

### For printed thesis:
- **Single-column figures**: width = 0.8\textwidth
- **Wide figures (scatter, complex)**: width = 0.95\textwidth or 1.0\textwidth
- **Small figures (diagrams)**: width = 0.6-0.7\textwidth

### Recommended widths for each figure:
- **Figure 5.1** (Horizontal bar chart): 0.8\textwidth
- **Figure 5.2** (Dual panel): 0.9\textwidth (needs space for both panels)
- **Figure 5.3** (Bar chart with error bars): 0.8\textwidth
- **Figure 5.4** (Learning curve): 0.9\textwidth (time series needs width)
- **Figure 5.5** (Distribution overlay): 0.9\textwidth
- **Figure 5.6** (Scatter plot): 0.85\textwidth (square aspect ratio)
- **Figure 5.7** (Conceptual diagram): 0.9\textwidth

---

## Quality Checklist Before Insertion

- [ ] All 7 PNG files are 300 DPI
- [ ] No embedded "Figure 5.X:" titles in images
- [ ] Axis labels are readable at final size
- [ ] Legends are clear and positioned well
- [ ] Colors are distinguishable (check grayscale if printing B&W)
- [ ] Captions are complete and accurate
- [ ] Cross-references in text match figure numbers
- [ ] Figure placement follows narrative flow

---

## Alternative Placement Options

If space is limited, consider:

1. **Combining related figures**: Group 5.1 and 5.6 (both about simulator validation) on facing pages
2. **Appendix placement**: Move Figure 5.7 (conceptual diagram) to appendix if referenced only once
3. **Page breaks**: Use `\clearpage` in LaTeX before each section to start figures on new pages

---

## Common LaTeX Issues & Solutions

**Issue**: Figure appears pages away from reference
**Solution**: Use `[htbp!]` or `\FloatBarrier` from `placeins` package

**Issue**: Figure too large for page
**Solution**: Reduce width or use `\resizebox{\textwidth}{!}{...}`

**Issue**: Caption text too wide
**Solution**: Use `\captionsetup{width=0.9\textwidth}` from `caption` package

**Issue**: Figures overlap with text
**Solution**: Adjust vertical spacing with `\vspace{1em}` before/after figure

---

## File Paths Reference

All figures are located in:
```
figures/figure5_1_simulator_validation.png
figures/figure5_2_action_masking_impact.png
figures/figure5_3_policy_performance.png
figures/figure5_4_learning_curve.png
figures/figure5_5_cost_distribution.png
figures/figure5_6_simulator_fidelity.png
figures/figure5_7_action_masking_mechanism.png
```

**Relative path from thesis root**: `figures/`
**Absolute path**: `/Users/marvin.schumann/.../thesis_private/figures/`

---

## Additional Resources

**LaTeX packages to include**:
```latex
\usepackage{graphicx}  % For \includegraphics
\usepackage{caption}   % For caption customization
\usepackage{subcaption}  % For subfigures (if needed)
\usepackage{float}     % For [H] placement
\usepackage{placeins}  % For \FloatBarrier
```

**Word tips**:
- Use "Insert Caption" not manual text to ensure auto-numbering
- Update field codes (Ctrl+A, then F9) after adding/removing figures
- Use "Cross-reference" feature for figure references in text

---

**Last updated**: 2025-12-11
**Status**: Ready for thesis insertion
**Contact**: See `FIGURE_CAPTIONS.txt` for caption text
