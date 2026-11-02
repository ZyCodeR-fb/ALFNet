# Codex 提示词（复制下方分隔线之间的全部内容，粘贴到 Codex 新对话，项目目录选 A:\projects\ALFNet）

---

You are helping me redraw Figure 1 (the model architecture figure) of my NeurIPS paper on ALFNet. The final figure will be rebuilt by hand in PowerPoint, so your job is to produce (1) a polished full-figure draft image that I use as a layout and style target, and (2) clean asset sheets of small modules that I can crop and paste into PowerPoint.

## Read first (all paths are relative to the repository root A:\projects\ALFNet)

1. `paper/latex_neurips_20261020/imgs/figure_design/FIG1_DESIGN_CN.md` — the authoritative design spec (palette, shapes, layout, curves, partition module, and the label correction table). Follow it exactly; where this prompt and the spec disagree, the spec wins.
2. `paper/latex_neurips_20261020/sections/method.tex` — the method section; every label and symbol in the figure must match it.
3. `paper/latex_neurips_20261020/imgs/figure_design/current_ALFNet_render.png` — the current figure (content is right, style and several labels are wrong).
4. Style references: `paper/latex_neurips_20261020/imgs/Reference_Image/1.png` … `7.png`. Use 3.png, 4.png and 7.png for the overall panel and tensor style (pastel rounded panels, small 3D cube tensors, rounded-rectangle layer boxes, circled operators, bottom legend bar), 1.png panel (c) for the spatial-partition blobs, 1.png panel (a), 6.png panel (a) and 7.png's decoder output boxes for the curve style, and 4.png for stars, triangles and bar-chart icons.

## Skills

Use the installed `nature-figure` skill, route “AI-generated mechanism schematic”, with the built-in image generation tool. Important override: do NOT apply the Nature minimal vector-schematic style rules (no-3D, no-gradient, two-colour limits). This is a NeurIPS / KDD-style PowerPoint architecture figure and must look like the reference images: white canvas, pastel module panels with coloured borders, small 3D cube tensors, flat icons. Keep the spec's data-colour rule: the same variable has the same colour everywhere.

Do not use any other skill unless I ask. Do not edit, move or delete any existing file. Write outputs only to `paper/latex_neurips_20261020/imgs/figure_design/codex_out/`. Do not run git commands.

## Deliverables (in codex_out/)

1. `fig1_draft_v1.png` — the full architecture figure, aspect ratio about 2:1 (landscape), at least 3000 px wide, white background. Generate 2 variants (`fig1_draft_v1a.png`, `fig1_draft_v1b.png`) if the first has layout errors.
2. `assets_curves.png` — the three curve panels described in section 3 of the spec, each on its own white tile with generous margins, no text except tiny axis ticks.
3. `assets_modules.png` — a sheet of separate small modules on a white background with clear spacing: (a) the spatial-partition module (road graph with 3–4 translucent partition blobs + the P×C slot grid with one hatched padded slot); (b) four teal latent-token stars with gather/broadcast arrows to a row of slots; (c) a router icon: “Linear → Softmax” with a 5-bar weight chart (five different pastel colours, one per expert); (d) five small 6×6 adjacency-matrix heatmap icons for experts P₁…P₅ (identity diagonal, binary, weighted, forward, backward patterns); (e) a FiLM “bottleneck” head icon (wide–narrow–wide trapezoids) with two output bars Δγ (rose) and β (light rose); (f) a temporal-convolution trapezoid compressing L time steps into one patch; (g) role glyphs: AAS solid blue circle, NAS green triangle, VS grey dashed hollow circle, FS red cross; (h) a two-layer graph-GRU cell unrolled over time with dashed purple arrows entering each cell from above.
4. `README.md` — the exact prompts you used, the image tool/model name, which reference images you passed as inputs, and a checklist confirming every item of the spec's section 5 label table appears correctly (or listing what the image model rendered wrongly, so I retype it in PowerPoint).

## Image prompt for fig1_draft (use as the main prompt; pass 3.png, 4.png and current_ALFNet_render.png as reference images)

A clean, professional machine-learning model architecture diagram for a NeurIPS paper, landscape 2:1, white background, PowerPoint style like the reference figures: pastel rounded panels with thin coloured borders, small isometric 3D cube tensors, rounded-rectangle layer boxes, circled operators (⊕ ⊙), thin black orthogonal arrows, a legend bar at the bottom. Flat icons, no photorealism, no drop shadows, crisp vector look, consistent serif font, all text horizontal and legible.

Layout from left to right:
- Left column “Inputs”, three stacked panels: (1) “Visible traffic history X” with three thin swim-lanes of smooth blue curves on a shared time axis 1…L: lane 1 a full blue curve with one hollow circle (natural missing), lane 2 grey dotted line then a blue curve for the last steps starting at a green triangle (newly added sensor), lane 3 a blue curve that stops at a red cross followed by a grey dotted line (failed sensor); (2) purple panel “Lifecycle context S = [R, a, M]” showing a strip of four-colour role squares, a purple staircase bar chart labelled a_t, and a black-and-white checker row labelled M_t, plus four short trajectory bars labelled Unseen, Steady, Join, Fault; (3) orange panel “Static graph A” with a small node-edge road graph.
- Second column, coral-bordered panel “SAHC”, top to bottom: “Bidirectional graph recurrent completion” with two rows of tiny graph-GRU cells (“Forward 1→L”, “Backward L→1”) merging into an “MLP fusion” box, output a blue cube slab labelled X^c; then “Window interpolation (ι)” with one curve whose missing points (hollow circles) are joined by dashed straight segments; then “Reference centering” with a circled minus and a yellow box “αv”; output a blue cube slab labelled X̄ that splits into two arrows to the right.
- Top-right, green-bordered panel “ALFTemporal”, left to right: stacked inputs (blue X̄ cubes, purple S cubes, gold clock and calendar icons for τ, δ) entering a trapezoid “Temporal conv (kernel = stride = L)”, output a short teal bar e_n; a long concatenated vector bar coloured teal | gold | gold | orange labelled z⁰ (160); a “Spatial partitions” module: a light road graph with three translucent coloured blobs and, next to it, a coloured P×C slot grid with one hatched cell; a “Two-stage attention” module: stage 1 highlights one column of the grid (“cross-partition”), stage 2 highlights one row connected to four teal stars (“within-partition, 4 latent tokens”) with gather and broadcast arrows; a teal bar stack “Z (N×160)” into a small bottleneck box “FiLM head”, outputs two rose bars Δγ and β.
- Bottom-right, blue-bordered panel “ALFGraph”, left to right: “Graph encoder” (two stacked rows of graph-GRU cells unrolled over t = 1…L, dashed purple arrows labelled S_t entering from above, teal hidden-state bars h⁽¹⁾_L, h⁽²⁾_L); a yellow “FiLM” box with ⊙ and ⊕ receiving Δγ and β from the ALFTemporal panel (one long arrow down); “Residual AR decoder” unrolled over k = 1…H with a feedback arrow y_{k−1} and a dashed purple S_L arrow, plus a thin grey dotted arrow labelled “Teacher Forcing (train)”; a small “Restore +αv” box; a peach “Role-scaled graph refinement” stack (inputs ỹ_k, P⁺ỹ_k, P⁻ỹ_k → Linear–GELU–Linear → tanh → ×0.05 s(R_L) → ⊕); final output panel “Ŷ” showing a short solid history curve, a vertical dashed line at t = L and dashed forecast curves.
- Below the encoder, a dashed zoom-in box “Graph GRU cell with OS-routed mixture of graph experts”: S_t → “Linear → Softmax” router with a five-bar weight chart w_t → five small matrix heatmaps P₁…P₅ (each times Θ_j) → circled Σ → G; on the right four small gate boxes r, u, h̃, h′.
- Bottom legend bar: ⊕ addition, ⊙ Hadamard product, solid arrow = data flow, dashed purple = OS conditioning, dotted grey = training only, role glyphs (AAS ●, NAS ▲, VS ◌, FS ✗), and colour key: blue traffic values, purple lifecycle context, gold calendar, orange graph structure, teal hidden states, rose FiLM parameters.

Colour palette (use exactly): blue #2E75B6 / #BDD7EE; purple #7F5FA8 / #E4DFEC; gold #BF9000 / #FFF2CC; orange #C55A11 / #FBE5D6; teal #31859C / #DAEEF3; rose #C0508A / #F2D0E3; green triangle #70AD47; red cross #C00000; panel backgrounds very light with borders coral #E8857A, sage #93C47D, steel blue #6FA8DC.

Do not include the words GRIN, BiGRIL, DCRNN or KD-tree anywhere. Do not add modules that are not in the spec.

## Image prompt for assets_curves.png

Three separate small chart tiles on a pure white background, flat vector style, matching reference figure 1 (a): smooth spline lines 2 px thick with a light 15% tinted area underneath, tiny filled circles at observed points, hollow circles at missing points, grey dotted segments where data is missing, no titles, only small axis ticks. Tile 1: three stacked swim-lanes as described for “Visible traffic history”. Tile 2: one interpolation curve, missing points joined by dashed straight segments, horizontal dashed extensions before the first and after the last observed point. Tile 3: forecast tile, a short solid blue history curve with a light blue fill, a vertical dashed line, then three slightly offset dashed forecast curves in blue shades.

## When done

Show me the images and the README checklist. If I ask afterwards, use the `nature-image2ppt` skill to reconstruct `fig1_draft_v1.png` as an editable PPTX in `codex_out/`.

---
