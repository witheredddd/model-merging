# Task-driven parameter substructure diagnostic

Status: COMPLETE; 488 physical parameter evaluations, 13 conditional deletion comparisons.

All scalar outcomes: `all_metrics.csv`; all class counts and metrics: `per_class_part*.csv.gz`; conditional objective gaps and residual certificates: `deletions.csv`; every native region: `regions.csv`; every resolved rank: `rank_curves.csv`; signed sampled compatibility: `compatibility.csv`. Raw JSON/NPZ retain paired per-image observations.

Basis construction (p0), control selection (p2), final evaluation (p3) are disjoint image sets. Probes are training-side development data, not an official test set and not claimed unseen during expert training. Rank is bounded by the finite demand family and is not intrinsic dimension. Fixed physical radius comparisons are direction tests, not fixed-response protection optima. Fisher deletion gaps concern the quadratic proxy, not CE. Compatibility is sampled, not a globally certified nonlinear Pareto frontier.

No new controlled expert was trained in this diagnostic, so Val >=80% and EC search superiority remain untested. Negative, zero and uncertain outcomes are retained. Source tests and a successful finite step are not evidence of necessary groups or a manifold.

## Fixed-radius sufficiency and full-complement controls

| Anchor | Demand | Family | Method | DTD weighted CE change | Approx 95% interval | DTD Top-1 change | Cars CE change | Cars Top-1 change | Held-out Log/JVP error |
|---|---|---|---|---:|---|---:|---:|---:|---:|
|0|cos_4|sufficiency|existing_fisher_rank1|-0.00960435|[-0.0225301, 0.00332138]|0.0212766|0.00531452|0.00961538|0.695156|
|0|cos_4|sufficiency|existing_fisher_rank2|-0.00958418|[-0.0225014, 0.00333306]|0.0212766|0.00528351|0.00961538|0.695019|
|0|cos_4|sufficiency|existing_fisher_rank3|-0.00946787|[-0.0223432, 0.00340748]|0.0212766|0.00515293|0.00961538|0.69051|
|0|cos_4|sufficiency|existing_fisher_rank4|-0.00945784|[-0.0223185, 0.00340283]|0.0212766|0.00523254|0.00961538|0.690086|
|0|cos_4|sufficiency|frozen_gradient_rank1|-0.00304464|[-0.00489852, -0.00119075]|0|7.72681e-05|0|0.11435|
|0|cos_4|sufficiency|frozen_gradient_rank2|-0.00588033|[-0.0104584, -0.00130231]|0|0.00092859|0|0.438621|
|0|cos_4|sufficiency|frozen_gradient_rank3|-0.00659855|[-0.011763, -0.00143408]|0.0106383|0.00142267|0|0.469837|
|0|cos_4|sufficiency|frozen_gradient_rank4|-0.00662962|[-0.0118165, -0.00144269]|0.0106383|0.00147925|0|0.471721|
|0|cos_4|sufficiency|frozen_gradient_rank5|-0.00667415|[-0.0119067, -0.00144163]|0.0106383|0.00151074|0|0.477007|
|0|cos_4|sufficiency|frozen_gradient_rank6|-0.00695371|[-0.0122463, -0.00166117]|0.0106383|0.00138269|0|0.486766|
|0|cos_4|sufficiency|frozen_gradient_rank7|-0.00721249|[-0.0126645, -0.00176048]|0.0106383|0.00160074|0|0.498945|
|0|cos_4|sufficiency|frozen_gradient_rank8|-0.00715888|[-0.0128802, -0.00143759]|0.0106383|0.0017461|0|0.499177|
|0|cos_4|sufficiency|full_gradient|-0.00987184|[-0.017688, -0.00205572]|0.0106383|0.00237068|0|0.520923|
|0|cos_4|sufficiency|mean_gradient_1d|-0.00287659|[-0.00458465, -0.00116852]|0|1.15623e-06|0|0.121981|
|0|cos_4|motion_complement|outside_existing_fisher_rank1|-0.00349808|[-0.00540292, -0.00159324]|0|-0.000427645|0|0.16659|
|0|cos_4|motion_complement|outside_existing_fisher_rank2|-0.00349024|[-0.00538423, -0.00159625]|0|-0.00040941|0|0.166098|
|0|cos_4|motion_complement|outside_existing_fisher_rank3|-0.00363017|[-0.00555601, -0.00170432]|0|-0.000331813|0|0.172345|
|0|cos_4|motion_complement|outside_existing_fisher_rank4|-0.00363052|[-0.00555149, -0.00170955]|0|-0.000375393|0|0.171573|
|0|cos_4|motion_complement|outside_frozen_gradient_rank1|-0.00939637|[-0.0171619, -0.00163085]|0.0106383|0.0024278|0|0.52415|
|0|cos_4|motion_complement|outside_frozen_gradient_rank2|-0.00881347|[-0.0156283, -0.00199869]|0.0106383|0.00172927|0|0.462021|
|0|cos_4|motion_complement|outside_frozen_gradient_rank3|-0.00866919|[-0.0152665, -0.00207183]|0.0106383|0.00153155|0|0.446587|
|0|cos_4|motion_complement|outside_frozen_gradient_rank4|-0.008655|[-0.0152432, -0.00206681]|0.0106383|0.00151108|0|0.446551|
|0|cos_4|motion_complement|outside_frozen_gradient_rank5|-0.00864679|[-0.0152183, -0.00207532]|0.0106383|0.00150012|0|0.444487|
|0|cos_4|motion_complement|outside_frozen_gradient_rank6|-0.0085381|[-0.0150559, -0.00202035]|0.0106383|0.00151803|0|0.438691|
|0|cos_4|motion_complement|outside_frozen_gradient_rank7|-0.00845104|[-0.0148933, -0.00200876]|0.0106383|0.00143884|0|0.431692|
|0|cos_4|motion_complement|outside_frozen_gradient_rank8|-0.00844336|[-0.0147627, -0.00212403]|0.0106383|0.00136981|0|0.426242|
|0|mean|sufficiency|existing_fisher_rank1|-0.00710976|[-0.0164319, 0.00221243]|0.0212766|0.00531452|0.00961538|0.695156|
|0|mean|sufficiency|existing_fisher_rank2|-0.00712147|[-0.0164343, 0.00219132]|0.0212766|0.00536715|0.00961538|0.695263|
|0|mean|sufficiency|existing_fisher_rank3|-0.00712989|[-0.0164434, 0.00218357]|0.0212766|0.00537497|0.00961538|0.695517|
|0|mean|sufficiency|existing_fisher_rank4|-0.00713202|[-0.0164799, 0.00221582]|0.0212766|0.00509749|0.00961538|0.696427|
|0|mean|euclidean_freeze|matched_contiguous_control|-0.00780744|[-0.0137053, -0.00190957]|0.0106383|0.00280132|0.00961538|0.554472|
|0|mean|euclidean_freeze|signed_conflict_union|-0.00208379|[-0.00307673, -0.00109084]|0|1.20869e-05|0|0.00428204|
|0|mean|euclidean_freeze|top_native_region|-0.00265858|[-0.00382734, -0.00148982]|0|2.60466e-05|0|0.00419806|
|0|mean|sufficiency|frozen_gradient_rank1|-0.00222013|[-0.00364451, -0.000795751]|0|7.72681e-05|0|0.11435|
|0|mean|sufficiency|frozen_gradient_rank2|-0.00449249|[-0.00779576, -0.00118922]|0|0.000930814|0|0.439069|
|0|mean|sufficiency|frozen_gradient_rank3|-0.00547363|[-0.00931756, -0.00162971]|0.0106383|0.00160007|0|0.476393|
|0|mean|sufficiency|frozen_gradient_rank4|-0.00548763|[-0.00934389, -0.00163136]|0.0106383|0.00163845|0|0.477749|
|0|mean|sufficiency|frozen_gradient_rank5|-0.00557774|[-0.00951688, -0.0016386]|0.0106383|0.00171137|0|0.490181|
|0|mean|sufficiency|frozen_gradient_rank6|-0.00576222|[-0.00974739, -0.00177705]|0.0106383|0.00159637|0|0.498954|
|0|mean|sufficiency|frozen_gradient_rank7|-0.00600543|[-0.0101347, -0.00187616]|0.0106383|0.00182996|0|0.510278|
|0|mean|sufficiency|frozen_gradient_rank8|-0.00594613|[-0.0102057, -0.00168654]|0.0106383|0.00193416|0.00961538|0.512585|
|0|mean|sufficiency|full_gradient|-0.00780961|[-0.0136915, -0.00192773]|0.0106383|0.00278813|0.00961538|0.553567|
|0|mean|sufficiency|mean_gradient_1d|-0.00216578|[-0.00354068, -0.000790881]|0|1.15623e-06|0|0.121981|
|0|mean|motion_complement|outside_existing_fisher_rank1|-0.00224264|[-0.00335603, -0.00112924]|0|-0.000251571|0|0.147014|
|0|mean|motion_complement|outside_existing_fisher_rank2|-0.00227196|[-0.00340991, -0.00113401]|0|-0.000289118|0|0.16946|
|0|mean|motion_complement|outside_existing_fisher_rank3|-0.0022611|[-0.00339797, -0.00112423]|0|-0.000295043|0|0.175875|
|0|mean|motion_complement|outside_existing_fisher_rank4|-0.00228564|[-0.00342335, -0.00114793]|0|-0.000138355|0|0.173729|
|0|mean|motion_complement|outside_frozen_gradient_rank1|-0.0074701|[-0.0133528, -0.00158734]|0.0106383|0.00284349|0.00961538|0.555709|
|0|mean|motion_complement|outside_frozen_gradient_rank2|-0.00712099|[-0.0123538, -0.00188817]|0.0106383|0.0022511|0|0.509498|
|0|mean|motion_complement|outside_frozen_gradient_rank3|-0.0069008|[-0.0119011, -0.00190049]|0.0106383|0.00195282|0|0.492605|
|0|mean|motion_complement|outside_frozen_gradient_rank4|-0.00689325|[-0.0118888, -0.00189774]|0.0106383|0.00193638|0|0.492627|
|0|mean|motion_complement|outside_frozen_gradient_rank5|-0.00687066|[-0.0118225, -0.00191884]|0.0106383|0.00189885|0|0.487001|
|0|mean|motion_complement|outside_frozen_gradient_rank6|-0.0067975|[-0.0117126, -0.00188236]|0.0106383|0.00191531|0|0.482194|
|0|mean|motion_complement|outside_frozen_gradient_rank7|-0.00670501|[-0.0115495, -0.00186055]|0.0106383|0.00180157|0|0.47532|
|0|mean|motion_complement|outside_frozen_gradient_rank8|-0.00672363|[-0.0115139, -0.00193335]|0.0106383|0.00175147|0|0.47234|
|0|sin_4|sufficiency|existing_fisher_rank1|-0.00884049|[-0.0194021, 0.00172108]|0.0212766|0.00531452|0.00961538|0.695156|
|0|sin_4|sufficiency|existing_fisher_rank2|-0.00885074|[-0.0193706, 0.00166916]|0.0212766|0.00547735|0.00961538|0.694733|
|0|sin_4|sufficiency|existing_fisher_rank3|-0.00901391|[-0.0195464, 0.00151858]|0.0212766|0.00563454|0.00961538|0.699362|
|0|sin_4|sufficiency|existing_fisher_rank4|-0.00903672|[-0.0195624, 0.00148893]|0.0212766|0.00571774|0.00961538|0.698925|
|0|sin_4|euclidean_freeze|matched_contiguous_control|-0.00642206|[-0.0105083, -0.00233585]|0.0106383|0.001626|0|0.376291|
|0|sin_4|euclidean_freeze|signed_conflict_union|-0.00174731|[-0.0025876, -0.000907016]|0|1.69638e-05|0|0.00373897|
|0|sin_4|euclidean_freeze|top_native_region|-0.00230711|[-0.00333589, -0.00127833]|0|2.40801e-05|0|0.00380594|
|0|sin_4|sufficiency|frozen_gradient_rank1|-0.00161911|[-0.00316999, -6.82239e-05]|0|7.72681e-05|0|0.11435|
|0|sin_4|sufficiency|frozen_gradient_rank2|-0.00445437|[-0.00770406, -0.00120467]|0|0.000773014|0|0.400479|
|0|sin_4|sufficiency|frozen_gradient_rank3|-0.00565351|[-0.00965474, -0.00165228]|0.0106383|0.00137203|0|0.438952|
|0|sin_4|sufficiency|frozen_gradient_rank4|-0.00565045|[-0.00964866, -0.00165223]|0.0106383|0.00136296|0|0.438398|
|0|sin_4|sufficiency|frozen_gradient_rank5|-0.0056897|[-0.00979618, -0.00158322]|0.0106383|0.0014326|0|0.454008|
|0|sin_4|sufficiency|frozen_gradient_rank6|-0.00591655|[-0.0100989, -0.00173424]|0.0106383|0.00128495|0|0.465099|
|0|sin_4|sufficiency|frozen_gradient_rank7|-0.00631661|[-0.0107453, -0.00188787]|0.0106383|0.00154238|0|0.480718|
|0|sin_4|sufficiency|frozen_gradient_rank8|-0.00636142|[-0.0110019, -0.00172095]|0.0106383|0.00164157|0|0.484408|
|0|sin_4|sufficiency|full_gradient|-0.00640713|[-0.0104719, -0.00234231]|0.0106383|0.00161691|0|0.374021|
|0|sin_4|sufficiency|mean_gradient_1d|-0.00147902|[-0.00299085, 3.28081e-05]|0|1.15623e-06|0|0.121981|
|0|sin_4|motion_complement|outside_existing_fisher_rank1|-0.00230054|[-0.00354724, -0.00105385]|0|0.000354507|0|0.099851|
|0|sin_4|motion_complement|outside_existing_fisher_rank2|-0.0023582|[-0.00361833, -0.00109806]|0|0.000286263|0|0.0921716|
|0|sin_4|motion_complement|outside_existing_fisher_rank3|-0.00226123|[-0.00346645, -0.00105602]|0|0.000209087|0|0.0937774|
|0|sin_4|motion_complement|outside_existing_fisher_rank4|-0.002256|[-0.0034611, -0.00105091]|0|0.000183249|0|0.0947754|
|0|sin_4|motion_complement|outside_frozen_gradient_rank1|-0.00620467|[-0.010245, -0.00216431]|0.0106383|0.00164454|0|0.37663|
|0|sin_4|motion_complement|outside_frozen_gradient_rank2|-0.0055314|[-0.00894138, -0.00212143]|0.0106383|0.00138078|0|0.303589|
|0|sin_4|motion_complement|outside_frozen_gradient_rank3|-0.005235|[-0.00839789, -0.00207211]|0.0106383|0.00124599|0|0.285832|
|0|sin_4|motion_complement|outside_frozen_gradient_rank4|-0.00523504|[-0.00839852, -0.00207156]|0.0106383|0.0012478|0|0.285859|
|0|sin_4|motion_complement|outside_frozen_gradient_rank5|-0.00520158|[-0.00832014, -0.00208301]|0.0106383|0.00123425|0|0.280267|
|0|sin_4|motion_complement|outside_frozen_gradient_rank6|-0.00512633|[-0.00820593, -0.00204673]|0.0106383|0.00126144|0|0.275082|
|0|sin_4|motion_complement|outside_frozen_gradient_rank7|-0.00498796|[-0.00796292, -0.002013]|0|0.00119403|0|0.266333|
|0|sin_4|motion_complement|outside_frozen_gradient_rank8|-0.00496479|[-0.00788441, -0.00204517]|0|0.00117037|0|0.263558|

## Conditional proxy gaps and numerical certificates

| Case | Proxy deletion | Lower | Upper | Compensation proxy gain | Restricted residual |
|---|---:|---:|---:|---:|---:|
|a0_mean_p0_alpha0.1_freeze_learned_motion|370.741|370.741|370.741|2.94477|unresolved|
|a0_mean_p0_alpha0.1_signed_conflict_union|194.449|194.449|194.449|0.417994|4.50527e-06|
|a0_mean_p0_alpha0.1_top_native_region|110.264|110.264|110.264|0.288692|7.09466e-07|
|a0_mean_p0_alpha1.0_freeze_learned_motion|247.639|247.639|247.639|53.7002|unresolved|
|a0_mean_p0_alpha1.0_signed_conflict_union|109.903|109.903|109.903|5.37719|4.23106e-05|
|a0_mean_p0_alpha1.0_top_native_region|57.2015|57.2015|57.2015|3.57169|6.79745e-05|
|a0_mean_p0_alpha10.0_freeze_learned_motion|102.046|102.046|102.046|180.594|unresolved|
|a0_mean_p0_alpha10.0_signed_conflict_union|34.1231|34.1231|34.1234|7.57835|0.000442675|
|a0_mean_p0_alpha10.0_top_native_region|14.5728|14.5727|14.5731|5.31706|0.000587495|
|a0_sin_4_p2_alpha1.0_freeze_learned_motion|11.9777|11.9777|11.9778|3.12035|0.000147805|
|a0_sin_4_p2_alpha1.0_matched_contiguous_control|7.49725|7.49721|7.49729|0.0140406|0.000193514|
|a0_sin_4_p2_alpha1.0_signed_conflict_union|410.606|410.606|410.606|5.43001|2.93217e-05|
|a0_sin_4_p2_alpha1.0_top_native_region|246.243|246.243|246.243|6.44398|5.55529e-05|

Freezing the span used to construct a training gradient can trivially remove that same gradient. This is a construction identity, not evidence of irreplaceable physical parameters. Independent demands, independent images, later positions and actual finite outcomes must support any stronger interpretation. The arithmetic-resolution test is a numerical safeguard, not a statistical discovery threshold.

Fisher comparisons include common-objective, budget-feasible native scaling as well as individually normalized direction probes. These are distinct experiments: use `fisher_common_objective` to compare the same quadratic objective, and `fisher_freeze` only to compare directions at equal parameter norm. The unscaled quadratic optimum need not lie in a region where the nonlinear CE approximation is accurate; inspect recorded finite response errors. Approximate paired intervals do not include a multiple-comparison correction.

Numerical residual bounds and deletion intervals are algebraic bounds for an ideal fixed Jacobian and exact projector. Floating-point operator error, uncertainty in data sampling and error of the nonlinear quadratic approximation are not enclosed by these bounds. The projected-gradient arithmetic floor is a conservative numerical heuristic, not a formally verified interval.

`finite_substitution_gaps.csv` reports paired real finite CE/Top-1 differences between full and reoptimized-complement actions at the same objective and radius bound, with approximate paired intervals. These are outcomes of the selected proxy-optimal actions, not globally optimal nonlinear CE deletion gaps. The small pilot final Cars probe covers only 104 of 196 classes; it cannot represent a complete Cars evaluation. The expanded probes cover every class.
