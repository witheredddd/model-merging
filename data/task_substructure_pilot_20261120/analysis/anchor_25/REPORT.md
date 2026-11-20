# Task-driven parameter substructure diagnostic

Status: COMPLETE; 432 physical parameter evaluations, 0 conditional deletion comparisons.

All scalar outcomes: `all_metrics.csv`; all class counts and metrics: `per_class_part*.csv.gz`; conditional objective gaps and residual certificates: `deletions.csv`; every native region: `regions.csv`; every resolved rank: `rank_curves.csv`; signed sampled compatibility: `compatibility.csv`. Raw JSON/NPZ retain paired per-image observations.

Basis construction (p0), control selection (p2), final evaluation (p3) are disjoint image sets. Probes are training-side development data, not an official test set and not claimed unseen during expert training. Rank is bounded by the finite demand family and is not intrinsic dimension. Fixed physical radius comparisons are direction tests, not fixed-response protection optima. Fisher deletion gaps concern the quadratic proxy, not CE. Compatibility is sampled, not a globally certified nonlinear Pareto frontier.

No new controlled expert was trained in this diagnostic, so Val >=80% and EC search superiority remain untested. Negative, zero and uncertain outcomes are retained. Source tests and a successful finite step are not evidence of necessary groups or a manifold.

## Fixed-radius sufficiency and full-complement controls

| Anchor | Demand | Family | Method | DTD weighted CE change | Approx 95% interval | DTD Top-1 change | Cars CE change | Cars Top-1 change | Held-out Log/JVP error |
|---|---|---|---|---:|---|---:|---:|---:|---:|
|25|cos_4|sufficiency|existing_fisher_rank1|-0.00876665|[-0.021689, 0.00415573]|0|0.0049272|0.00961538|0.671433|
|25|cos_4|sufficiency|existing_fisher_rank2|-0.00876787|[-0.0216903, 0.00415457]|0|0.00493227|0.00961538|0.671448|
|25|cos_4|sufficiency|existing_fisher_rank3|-0.00851648|[-0.0213075, 0.00427456]|0|0.00476081|0.00961538|0.665039|
|25|cos_4|sufficiency|existing_fisher_rank4|-0.00839972|[-0.0210507, 0.00425126]|0|0.0052195|0.00961538|0.661562|
|25|cos_4|sufficiency|frozen_gradient_rank1|-0.00247343|[-0.00423999, -0.000706863]|0|9.28265e-05|0|0.126293|
|25|cos_4|sufficiency|frozen_gradient_rank2|-0.00530432|[-0.0101616, -0.000447012]|0.0106383|0.000974388|0.00961538|0.414189|
|25|cos_4|sufficiency|frozen_gradient_rank3|-0.00589866|[-0.0113071, -0.000490206]|0.0106383|0.00142418|0.00961538|0.456032|
|25|cos_4|sufficiency|frozen_gradient_rank4|-0.0059013|[-0.0113144, -0.000488167]|0.0106383|0.00143122|0.00961538|0.456316|
|25|cos_4|sufficiency|frozen_gradient_rank5|-0.00590818|[-0.0113266, -0.000489711]|0.0106383|0.00143564|0.00961538|0.45717|
|25|cos_4|sufficiency|frozen_gradient_rank6|-0.00617529|[-0.0116698, -0.000680769]|0.0106383|0.0013197|0.00961538|0.467088|
|25|cos_4|sufficiency|frozen_gradient_rank7|-0.00639458|[-0.0120426, -0.000746604]|0.0106383|0.00153058|0.00961538|0.479364|
|25|cos_4|sufficiency|frozen_gradient_rank8|-0.00635842|[-0.0123015, -0.00041533]|0.0106383|0.00168505|0.00961538|0.47532|
|25|cos_4|sufficiency|full_gradient|-0.00817839|[-0.0150771, -0.00127964]|0.0106383|0.00161147|0.00961538|0.460783|
|25|cos_4|sufficiency|mean_gradient_1d|-0.00225603|[-0.00386052, -0.00065154]|0|2.66396e-05|0|0.11523|
|25|cos_4|motion_complement|outside_existing_fisher_rank1|-0.00260301|[-0.00445249, -0.000753538]|0.0106383|-0.000472523|0|0.192603|
|25|cos_4|motion_complement|outside_existing_fisher_rank2|-0.0026023|[-0.00445224, -0.000752367]|0.0106383|-0.000475478|0|0.192521|
|25|cos_4|motion_complement|outside_existing_fisher_rank3|-0.00277651|[-0.00464672, -0.000906286]|0.0106383|-0.000378178|0|0.110127|
|25|cos_4|motion_complement|outside_existing_fisher_rank4|-0.00278563|[-0.00464629, -0.000924972]|0.0106383|-0.000626077|0|0.111103|
|25|cos_4|motion_complement|outside_frozen_gradient_rank1|-0.00788734|[-0.0147157, -0.00105898]|0.0106383|0.00162664|0.00961538|0.462252|
|25|cos_4|motion_complement|outside_frozen_gradient_rank2|-0.0072377|[-0.0131317, -0.00134374]|0.0106383|0.00123729|0.00961538|0.392482|
|25|cos_4|motion_complement|outside_frozen_gradient_rank3|-0.00707172|[-0.0127844, -0.00135902]|0.0106383|0.00110434|0.00961538|0.378796|
|25|cos_4|motion_complement|outside_frozen_gradient_rank4|-0.00707096|[-0.0127821, -0.00135982]|0.0106383|0.00110229|0.00961538|0.378745|
|25|cos_4|motion_complement|outside_frozen_gradient_rank5|-0.00706906|[-0.0127785, -0.00135965]|0.0106383|0.00110121|0.00961538|0.378484|
|25|cos_4|motion_complement|outside_frozen_gradient_rank6|-0.00696822|[-0.0126235, -0.00131293]|0.0106383|0.00111826|0|0.373307|
|25|cos_4|motion_complement|outside_frozen_gradient_rank7|-0.00687028|[-0.012445, -0.00129553]|0.0106383|0.00104716|0|0.365939|
|25|cos_4|motion_complement|outside_frozen_gradient_rank8|-0.00680546|[-0.0122158, -0.00139515]|0.0106383|0.000972265|0|0.358996|
|25|cos_4|sufficiency|refreshed_gradient_rank1|-0.0022046|[-0.00381699, -0.000592211]|0|-3.35931e-06|0|0.146176|
|25|cos_4|sufficiency|refreshed_gradient_rank2|-0.00562763|[-0.0108814, -0.000373894]|0.0106383|0.00101657|0.00961538|0.449172|
|25|cos_4|sufficiency|refreshed_gradient_rank3|-0.00608576|[-0.0118027, -0.000368856]|0.0106383|0.00139802|0.00961538|0.479685|
|25|cos_4|sufficiency|refreshed_gradient_rank4|-0.00610445|[-0.011825, -0.000383896]|0.0106383|0.00136473|0.00961538|0.479731|
|25|cos_4|sufficiency|refreshed_gradient_rank5|-0.00636243|[-0.0121648, -0.000560039]|0.0106383|0.00127507|0.00961538|0.485334|
|25|cos_4|sufficiency|refreshed_gradient_rank6|-0.00642748|[-0.0122738, -0.000581207]|0.0106383|0.00129329|0.00961538|0.491326|
|25|cos_4|sufficiency|refreshed_gradient_rank7|-0.00666615|[-0.0126635, -0.000668839]|0.0106383|0.00158991|0.00961538|0.503151|
|25|cos_4|sufficiency|refreshed_gradient_rank8|-0.00665019|[-0.012848, -0.000452378]|0.0106383|0.00176679|0.00961538|0.498715|
|25|mean|sufficiency|existing_fisher_rank1|-0.00618489|[-0.0155233, 0.0031535]|0|0.0049272|0.00961538|0.671433|
|25|mean|sufficiency|existing_fisher_rank2|-0.0061816|[-0.0155156, 0.00315244]|0|0.00494361|0.00961538|0.671478|
|25|mean|sufficiency|existing_fisher_rank3|-0.00615881|[-0.0154891, 0.00317151]|0|0.00492765|0.00961538|0.670956|
|25|mean|sufficiency|existing_fisher_rank4|-0.00615672|[-0.0154295, 0.00311606]|0|0.00520652|0.00961538|0.669323|
|25|mean|euclidean_freeze|matched_contiguous_control|-0.00673956|[-0.0121301, -0.001349]|0.0106383|0.00200465|0.00961538|0.50213|
|25|mean|euclidean_freeze|signed_conflict_union|-0.00139185|[-0.00208692, -0.000696773]|0.0106383|3.63905e-05|0|0.00286416|
|25|mean|euclidean_freeze|top_native_region|-0.00190396|[-0.00293572, -0.000872206]|0.0106383|2.61166e-06|0|0.00368054|
|25|mean|sufficiency|frozen_gradient_rank1|-0.00172934|[-0.0030504, -0.000408278]|0|9.28265e-05|0|0.126293|
|25|mean|sufficiency|frozen_gradient_rank2|-0.00403596|[-0.00766139, -0.000410528]|0.0106383|0.00102685|0.00961538|0.428633|
|25|mean|sufficiency|frozen_gradient_rank3|-0.00498196|[-0.00916072, -0.000803201]|0.0106383|0.00174375|0.00961538|0.478603|
|25|mean|sufficiency|frozen_gradient_rank4|-0.00497999|[-0.0091489, -0.000811072]|0.0106383|0.00172594|0.00961538|0.477924|
|25|mean|sufficiency|frozen_gradient_rank5|-0.00508806|[-0.00930435, -0.000871775]|0.0106383|0.00179125|0.00961538|0.488466|
|25|mean|sufficiency|frozen_gradient_rank6|-0.00523645|[-0.0095049, -0.000967994]|0.0106383|0.00169035|0.00961538|0.495723|
|25|mean|sufficiency|frozen_gradient_rank7|-0.00546757|[-0.00987456, -0.00106057]|0.0106383|0.00191073|0.00961538|0.504045|
|25|mean|sufficiency|frozen_gradient_rank8|-0.00536847|[-0.00995188, -0.00078507]|0.0106383|0.00204209|0.00961538|0.503536|
|25|mean|sufficiency|full_gradient|-0.00673827|[-0.0121144, -0.00136213]|0.0106383|0.0019936|0.00961538|0.500918|
|25|mean|sufficiency|mean_gradient_1d|-0.00162583|[-0.00288034, -0.000371315]|0|2.66396e-05|0|0.11523|
|25|mean|motion_complement|outside_existing_fisher_rank1|-0.00193911|[-0.00313712, -0.000741103]|0.0106383|-0.000316854|0|0.112907|
|25|mean|motion_complement|outside_existing_fisher_rank2|-0.00194306|[-0.00314232, -0.000743806]|0.0106383|-0.000328036|0|0.112635|
|25|mean|motion_complement|outside_existing_fisher_rank3|-0.00196043|[-0.00316152, -0.000759336]|0.0106383|-0.000316829|0|0.111929|
|25|mean|motion_complement|outside_existing_fisher_rank4|-0.0019645|[-0.00317042, -0.000758573]|0.0106383|-0.000473907|0|0.104498|
|25|mean|motion_complement|outside_frozen_gradient_rank1|-0.00655221|[-0.0119063, -0.00119813]|0.0106383|0.00200955|0.00961538|0.50299|
|25|mean|motion_complement|outside_frozen_gradient_rank2|-0.00622898|[-0.0109332, -0.00152473]|0.0106383|0.00161979|0.00961538|0.446459|
|25|mean|motion_complement|outside_frozen_gradient_rank3|-0.00597504|[-0.0104541, -0.00149595]|0.0106383|0.00139027|0.00961538|0.426682|
|25|mean|motion_complement|outside_frozen_gradient_rank4|-0.00597569|[-0.0104582, -0.00149322]|0.0106383|0.00139577|0.00961538|0.426744|
|25|mean|motion_complement|outside_frozen_gradient_rank5|-0.00594335|[-0.0104001, -0.00148663]|0.0106383|0.00137585|0.00961538|0.422291|
|25|mean|motion_complement|outside_frozen_gradient_rank6|-0.00588235|[-0.0103029, -0.00146184]|0.0106383|0.00139532|0.00961538|0.417878|
|25|mean|motion_complement|outside_frozen_gradient_rank7|-0.00577641|[-0.010111, -0.00144185]|0.0106383|0.00130359|0.00961538|0.409745|
|25|mean|motion_complement|outside_frozen_gradient_rank8|-0.00577328|[-0.0100188, -0.00152771]|0.0106383|0.0012446|0.00961538|0.404804|
|25|mean|sufficiency|refreshed_gradient_rank1|-0.00146505|[-0.00285394, -7.61483e-05]|0|-3.35931e-06|0|0.146176|
|25|mean|sufficiency|refreshed_gradient_rank2|-0.00427116|[-0.00820025, -0.00034208]|0.0106383|0.00108101|0.00961538|0.460257|
|25|mean|sufficiency|refreshed_gradient_rank3|-0.00509778|[-0.00951788, -0.000677684]|0.0106383|0.00172601|0.00961538|0.496833|
|25|mean|sufficiency|refreshed_gradient_rank4|-0.00510945|[-0.00953069, -0.000688215]|0.0106383|0.00169372|0.00961538|0.496932|
|25|mean|sufficiency|refreshed_gradient_rank5|-0.00521701|[-0.00970416, -0.000729862]|0.0106383|0.00163005|0.00961538|0.50166|
|25|mean|sufficiency|refreshed_gradient_rank6|-0.00532126|[-0.00985185, -0.000790678]|0.0106383|0.00166061|0.00961538|0.510604|
|25|mean|sufficiency|refreshed_gradient_rank7|-0.00553973|[-0.0102112, -0.000868269]|0.0106383|0.0019992|0.00961538|0.520471|
|25|mean|sufficiency|refreshed_gradient_rank8|-0.00547177|[-0.0102468, -0.000696786]|0.0106383|0.00212881|0.00961538|0.519652|
|25|sin_4|sufficiency|existing_fisher_rank1|-0.00826413|[-0.0189076, 0.00237937]|0|0.0049272|0.00961538|0.671433|
|25|sin_4|sufficiency|existing_fisher_rank2|-0.00828065|[-0.0189284, 0.00236715]|0|0.00487247|0.00961538|0.671191|
|25|sin_4|sufficiency|existing_fisher_rank3|-0.00832051|[-0.0189789, 0.00233787]|0|0.00490597|0.00961538|0.67229|
|25|sin_4|sufficiency|existing_fisher_rank4|-0.0082991|[-0.0185665, 0.0019683]|0|0.00594603|0.00961538|0.659438|
|25|sin_4|euclidean_freeze|matched_contiguous_control|-0.00611978|[-0.0104231, -0.00181647]|0.0106383|0.00130683|0.00961538|0.3282|
|25|sin_4|euclidean_freeze|signed_conflict_union|-0.00122324|[-0.00182879, -0.000617696]|0.0106383|2.01759e-05|0|0.00247976|
|25|sin_4|euclidean_freeze|top_native_region|-0.00173441|[-0.00260449, -0.000864322]|0.0106383|-8.77933e-07|0|0.0033737|
|25|sin_4|sufficiency|frozen_gradient_rank1|-0.00124292|[-0.00272843, 0.000242587]|0|9.28265e-05|0|0.126293|
|25|sin_4|sufficiency|frozen_gradient_rank2|-0.00486737|[-0.00931869, -0.000416044]|0.0106383|0.00107449|0.00961538|0.441092|
|25|sin_4|sufficiency|frozen_gradient_rank3|-0.00610808|[-0.011335, -0.000881145]|0.0106383|0.00190634|0.00961538|0.487609|
|25|sin_4|sufficiency|frozen_gradient_rank4|-0.00606579|[-0.0112023, -0.000929319]|0.0106383|0.00176216|0.00961538|0.480124|
|25|sin_4|sufficiency|frozen_gradient_rank5|-0.006113|[-0.0112987, -0.000927313]|0.0106383|0.00181904|0.00961538|0.489158|
|25|sin_4|sufficiency|frozen_gradient_rank6|-0.00629868|[-0.011529, -0.00106839]|0.0106383|0.00167161|0.00961538|0.49618|
|25|sin_4|sufficiency|frozen_gradient_rank7|-0.00651164|[-0.0118398, -0.00118345]|0.0106383|0.00193713|0.00961538|0.498774|
|25|sin_4|sufficiency|frozen_gradient_rank8|-0.00647231|[-0.012089, -0.000855583]|0.0106383|0.00207173|0.00961538|0.494805|
|25|sin_4|sufficiency|full_gradient|-0.00610686|[-0.0103947, -0.00181898]|0.0106383|0.00130186|0|0.326673|
|25|sin_4|sufficiency|mean_gradient_1d|-0.00107379|[-0.00249269, 0.00034512]|0|2.66396e-05|0|0.11523|
|25|sin_4|motion_complement|outside_existing_fisher_rank1|-0.00257219|[-0.00465733, -0.00048706]|0.0106383|0.000179703|0|0.190644|
|25|sin_4|motion_complement|outside_existing_fisher_rank2|-0.0025633|[-0.00465052, -0.000476081]|0.0106383|0.000199151|0|0.189581|
|25|sin_4|motion_complement|outside_existing_fisher_rank3|-0.00254488|[-0.00462762, -0.000462149]|0.0106383|0.000184738|0|0.189697|
|25|sin_4|motion_complement|outside_existing_fisher_rank4|-0.00253901|[-0.00464373, -0.000434284]|0.0106383|-0.000150446|0|0.181189|
|25|sin_4|motion_complement|outside_frozen_gradient_rank1|-0.0060282|[-0.0102947, -0.00176166]|0.0106383|0.0013029|0|0.327439|
|25|sin_4|motion_complement|outside_frozen_gradient_rank2|-0.0053169|[-0.00898327, -0.00165054]|0.0106383|0.00109075|0|0.272839|
|25|sin_4|motion_complement|outside_frozen_gradient_rank3|-0.00501811|[-0.00850075, -0.00153548]|0.0106383|0.00094881|0|0.258631|
|25|sin_4|motion_complement|outside_frozen_gradient_rank4|-0.00501847|[-0.00851694, -0.00152]|0.0106383|0.000967617|0|0.258661|
|25|sin_4|motion_complement|outside_frozen_gradient_rank5|-0.00500101|[-0.00849132, -0.00151071]|0.0106383|0.000960417|0|0.256739|
|25|sin_4|motion_complement|outside_frozen_gradient_rank6|-0.00494092|[-0.00840947, -0.00147237]|0.0106383|0.000980068|0|0.254204|
|25|sin_4|motion_complement|outside_frozen_gradient_rank7|-0.00479509|[-0.00816109, -0.00142908]|0.0106383|0.000898957|0|0.247526|
|25|sin_4|motion_complement|outside_frozen_gradient_rank8|-0.00474202|[-0.00801537, -0.00146867]|0.0106383|0.000853771|0|0.243333|
|25|sin_4|sufficiency|refreshed_gradient_rank1|-0.000962375|[-0.00251746, 0.000592706]|0|-3.35931e-06|0|0.146176|
|25|sin_4|sufficiency|refreshed_gradient_rank2|-0.00509905|[-0.00994707, -0.000251024]|0.0106383|0.00117382|0.00961538|0.469357|
|25|sin_4|sufficiency|refreshed_gradient_rank3|-0.00621842|[-0.0117054, -0.000731452]|0.0106383|0.00196697|0.00961538|0.501026|
|25|sin_4|sufficiency|refreshed_gradient_rank4|-0.00625903|[-0.0117649, -0.000753122]|0.0106383|0.00191724|0.00961538|0.500972|
|25|sin_4|sufficiency|refreshed_gradient_rank5|-0.00641104|[-0.0119798, -0.000842315]|0.0106383|0.001832|0.00961538|0.505083|
|25|sin_4|sufficiency|refreshed_gradient_rank6|-0.00646726|[-0.0120764, -0.000858105]|0.0106383|0.00186285|0.00961538|0.513041|
|25|sin_4|sufficiency|refreshed_gradient_rank7|-0.0066914|[-0.0124795, -0.000903286]|0.0106383|0.0022588|0.00961538|0.522696|
|25|sin_4|sufficiency|refreshed_gradient_rank8|-0.00664393|[-0.0126072, -0.000680644]|0.0106383|0.00241072|0.00961538|0.51986|

## Conditional proxy gaps and numerical certificates

| Case | Proxy deletion | Lower | Upper | Compensation proxy gain | Restricted residual |
|---|---:|---:|---:|---:|---:|

Freezing the span used to construct a training gradient can trivially remove that same gradient. This is a construction identity, not evidence of irreplaceable physical parameters. Independent demands, independent images, later positions and actual finite outcomes must support any stronger interpretation. The arithmetic-resolution test is a numerical safeguard, not a statistical discovery threshold.

Fisher comparisons include common-objective, budget-feasible native scaling as well as individually normalized direction probes. These are distinct experiments: use `fisher_common_objective` to compare the same quadratic objective, and `fisher_freeze` only to compare directions at equal parameter norm. The unscaled quadratic optimum need not lie in a region where the nonlinear CE approximation is accurate; inspect recorded finite response errors. Approximate paired intervals do not include a multiple-comparison correction.

Numerical residual bounds and deletion intervals are algebraic bounds for an ideal fixed Jacobian and exact projector. Floating-point operator error, uncertainty in data sampling and error of the nonlinear quadratic approximation are not enclosed by these bounds. The projected-gradient arithmetic floor is a conservative numerical heuristic, not a formally verified interval.

`finite_substitution_gaps.csv` reports paired real finite CE/Top-1 differences between full and reoptimized-complement actions at the same objective and radius bound, with approximate paired intervals. These are outcomes of the selected proxy-optimal actions, not globally optimal nonlinear CE deletion gaps. The small pilot final Cars probe covers only 104 of 196 classes; it cannot represent a complete Cars evaluation. The expanded probes cover every class.
