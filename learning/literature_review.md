# Literature review

Om Mohan Pawar (25110218), ME 299, IIT Gandhinagar. Supervisor: Prof. Madhu Vadali.

This is the reading behind both parts of the project, kept from September to November 2026 and cleaned up for the
course record. Each entry says what the work does, how it relates to this project and what I took from it.
Venue, year and volume were checked against the publisher or arXiv page. Where I read only the abstract, the entry says so.

The review answers four questions, in the order I met them:

1. How are tendon-driven continuum robots (TDCRs) modelled, and which model fits a notched, modular robot? (§1)
2. How is shape error measured and reported? (§2)
3. Is the physics of my model new? (§3, a novelty check that changed the project)
4. Who else predicts shape without sensors, and what does learning add? (§4, §5)

---

## 1. Models of tendon-driven continuum robots

| Work | What it does | What I took from it |
|---|---|---|
| Webster & Jones, "Design and kinematic modeling of constant curvature continuum robots: a review", *IJRR* 29(13), 2010 | The constant-curvature framework; separates actuation → configuration from configuration → task space | My first baseline. On a loaded notched robot its error grows with the number of notches (about five-fold in Part A) and reaches 11 to 68 mm in Part B |
| Burgner-Kahrs, Rucker & Choset, "Continuum robots for medical applications: a survey", *IEEE T-RO* 31(6), 2015 | Survey of designs, models, sensing and control | Context and vocabulary; where notched-tube robots sit among designs |
| Jones & Walker, "Kinematics for multisection continuum robots", *IEEE T-RO* 22(1), 2006 | Standard actuator → shape → task mapping for many sections | The mapping I first used to command two sections. It fails when distal tendons load proximal sections (below) |
| Camarillo, Milne, Carlson, Zinn & Salisbury, "Mechanics modeling of tendon-driven continuum manipulators", *IEEE T-RO* 24(6), 2008 | Planar, single section: every cross-section carries a moment *Td* and a compression *T*; tendon displacement = bending + axial compression + tendon stretch | The foundation of the twin's tendon map, and of Part A's stroke budget (stroke = bending + compression). Its forward map assumes no slack |
| Camarillo, Carlson & Salisbury, "Configuration tracking for continuum manipulators with coupled tendon drive", *IEEE T-RO* 25(4), 2009 | 3D, multi-section: distal tendons load proximal sections; forward kinematics *with slack* as a convex complementarity problem | **Contains the pull-only multi-section model I had derived independently.** I verified numerically that its slack problem and my QP give the same configuration (to 3×10⁻⁶ on 20 poses). This moved my contribution from the model to identification and transfer |
| Rucker & Webster, "Statics and dynamics of continuum robots with general tendon routing and external loading", *IEEE T-RO* 27(6), 2011 | Cosserat-rod statics with tendons routed through the body | The independent reference solver against which my beam solver is checked (0.004 mm) |
| Oliver-Butler, Till & Rucker, "Continuum robot stiffness under external loads and prescribed tendon displacements", *IEEE T-RO* 35(2), 2019 | Prescribed tendon displacements with tension ≥ 0, slack ≥ 0 and complementarity, solved by shooting | The same complementarity in a full Cosserat model. Mine is its convex, beam-from-CAD counterpart: unique, no initial guess, milliseconds |
| Till, Aloi & Rucker, "Real-time dynamics of soft and continuum robots based on Cosserat rod models", *IJRR* 38(6), 2019 | Real-time Cosserat dynamics | The run-time budget a twin must meet; dynamics as a later step |
| Dalvand, Nahavandi & Howe, "An analytical loading model for n-tendon continuum robots", *IEEE T-RO* 34(5), 2018 | Which tendons carry load and which go slack for a given configuration | Background for pretension. The ratio 1 : 1.75 : 2.25 : 2.5 that keeps distal tendons taut was then computed with the twin |
| Gonthina, Wooten, Godage & Walker, "Mechanics for tendon actuated multisection continuum arms", *ICRA* 2020 | Multi-section mechanics with axial compression and recursive coupling; tension input, chosen to avoid slack | The tension-input counterpart of my displacement-input problem, where slack must be decided rather than avoided |
| Shen, Browne, Ha & Chen, "Friction modeling of tendon-driven continuum robots through linear complementarity problem", *IEEE RA-L*, 2026 | Capstan friction as an LCP | The friction term in my code (sub-millimetre effect in simulation); friction is the first suspect if hardware shows hysteresis |
| Cowper, "The shear coefficient in Timoshenko's beam theory", *J. Appl. Mech.* 33(2), 1966 | Shear coefficients for beam sections | The optional Timoshenko shear term |

## 2. Measuring and reporting error

| Work | What it does | What I took from it |
|---|---|---|
| Rao, Peyron, Lilge & Burgner-Kahrs, "How to model tendon-driven continuum robots and benchmark modelling performance", *Front. Robot. AI* 7, 2020 | Compares TDCR models on one benchmark; recommends tip error as a percentage of robot length and orientation error in degrees | Every error in the project is also reported as % of length. The hardware pass criterion (2 % of length, 13 mm for four modules) is set this way, before any data |

## 3. Simulation as ground truth

| Work | What it does | What I took from it |
|---|---|---|
| Faure et al., "SOFA: a multi-model framework for interactive physical simulation", Springer, 2012 | The SOFA framework | FEM ground truth for both parts (SOFA 25.06) |
| Coevoet et al., "Software toolkit for modeling, simulation, and control of soft robots", *Adv. Robot.* 31(22), 2017 | The SoftRobots plugin and its cable constraints | Its cable constraint is two-sided unless a minimum force is set: **simulated cables could push.** All scenes were changed to pull-only |
| Duriez, "Control of elastic soft robots based on real-time finite element method", *ICRA* 2013 | Real-time FEM from actuation to shape | Sensor-free real-time twins of soft robots are not new in themselves; the claim must be about transfer |
| Dubied, Michelis, Spielberg & Katzschmann, "Sim-to-real for soft robots using differentiable FEM", *IEEE RA-L*, 2022 | Recipes for meshing, damping and actuation so FEM matches hardware | Mesh and settling checks; material from measurement (TPU *E* = 56 MPa), not a guess |

**Lesson.** A simulator must be checked before it is trusted. In September I found cables that could push and gravity
1000 times too weak in our scenes, and runs logged at a step limit rather than at equilibrium. All three would have
gone into results unnoticed.

## 4. Shape estimation without sensors, and the closest work

| Work | What it does | Relation to this project |
|---|---|---|
| Lilge, Barfoot & Burgner-Kahrs, "Continuum robot state estimation using Gaussian process regression on SE(3)", *IJRR* 41(13–14), 2022 | GP over arc length fusing pose and strain measurements along the body | Uses sensors on the robot; the twin uses none. Its GP-over-arc-length idea shaped my first whole-body model |
| Wang & Rojas, "A data-efficient model-based learning framework for the closed-loop control of continuum robots", *RoboSoft* 2022 | GP learns the sim-to-real relation of the end effector | Tip only; I needed the whole body |
| Feliu-Talegon, Alkayas, Adamu, Mathew & Renda, "Dynamic shape estimation of tendon-driven soft manipulators via actuation readings", *IEEE RA-L* 10(1), 2025 | Variable-strain model and observer from tendon displacements and actuator forces | The sensor-free benchmark (about 7 % of length, dynamic). Uses actuator forces; the twin uses displacements only |
| Zhang, Yan, Du, Chen, Song & Cheng, "A general modular framework for modeling cable-driven continuum surgical robots", *Int. J. Mech. Sci.* 318, 2026 (read in full) | Modular *modelling components* across robot types; single section, planar, tension sensors | State of the art for notched single sections (about 2 % of length). Not stacked modules, no gravity or mounting change |
| Kim et al., "Digital twins for the design, interactive control, and deployment of modular, fiber-reinforced soft continuum arms", arXiv:2507.10121 (read in full) | Each pneumatic module calibrated alone, assemblies predicted to 2 to 8 % | **Closest work on composition.** There, actuation is local to a module. Here, tendons load every module below, so a module cannot be calibrated alone; the twin is identified on 1 and 2 modules instead |
| Kasaei, Alambeigi & Khadem, arXiv:2501.03859; Kasaei, Ghobadi & Khadem, arXiv:2510.12332 (abstracts) | Cosserat prior + neural-ODE residual for the whole shape, used for control | Physics + learned residual, one robot. My residual-learning experiments (below) showed why I moved away from residuals |

**Gap found (25 Oct 2026).** No work found calibrates on one or two sections of a TDCR and predicts longer stacks or
other mountings without sensors. That is the claim Part B tests.

## 5. Physics-informed learning (scientific machine learning)

| Work | What it does | Relation to this project |
|---|---|---|
| Feizi, Pedrosa, Patel & Jayender, "Few-shot physics-informed neural network for shape reconstruction of concentric-tube robots", *ICRA* 2026 (read in full) | Cosserat PINN from actuation only; 500 real observations; retrained for each design | Learns the *solution* of one robot |
| Danesh, Qarehdaghi & Janabi-Sharifi, "Constraint-aware physics-informed neural networks for static shape estimation of co-manipulative continuum robots", arXiv:2608.26273 (read in full) | PINN with equilibrium and loop-closure residuals; tendon displacement input; 0.5 mm after fine-tuning on 571 real poses | Same input as the twin; new lengths and layouts left to future work |
| Zhan et al., "Physics informed neural pose estimation for real-time shape reconstruction of soft continuum robots", *IEEE RA-L* 10(7), 2025 (read in full) | Shape from embedded sensors (FBG strains and IMUs) with exact-curvature kinematics | Sensor-based, so not a sensor-free competitor |
| Hong, Lee, Song & Park, "Bridging high-fidelity simulations and physics-based learning using a surrogate model for soft robot control", *Adv. Intell. Syst.*, 2025 (read in full) | SOFA calibrated to hardware, model reduction, Transformer + PINN surrogate | Simulation-to-surrogate, one robot and one configuration |
| Frieden, Ferguson, Kuntz & Shankar, "Neural operators for design-space surrogate modeling of tendon-actuated continuum robots", *ICRA* 2026 (abstract) | DeepONet / FNO from design parameters and tendon inputs to shape | Learning *across designs*: the tool for step 2 of my future plan |

**What I concluded.** Physics-informed, actuation-only shape estimation is established. Every one of these works learns
the solution map of one robot in one configuration and is retrained for a new design. So I learn the module's
*constitutive law* inside an exact equilibrium solver. The learned part then transfers to new stacks and mountings by
construction. I tested this against forward and inverse PINNs trained on the same data: they fit the training
configurations (3 to 6 mm) but not new mountings (9 to 77 mm) or four modules (54 to 73 mm, after retraining). The
identified law stays at 0.2 to 0.5 mm.

---

## How the reading changed the project

| When | What I read | What changed |
|---|---|---|
| Late Sep | Lilge 2022, Wang & Rojas 2022 | From a tip-only GP to a whole-backbone model (shape modes along the arc length) |
| 29 Sep | Camarillo 2008, Rucker & Webster 2011, Coevoet 2017 | Distal tendons load proximal sections; the simulator's cables could push → pull-only everywhere |
| 4–5 Oct | Camarillo 2009, Oliver-Butler 2019, Gonthina 2020 | The pull-only multi-section model is not new → contribution moved to identification, transfer and hardware |
| 25–26 Oct | Zhang 2026, Feliu-Talegon 2025, Kim 2025 | Positioned the twin as the first to calibrate small and deploy large without sensors |
| 29 Oct | Feizi 2026, Danesh 2026, Zhan 2025, Hong 2025 | "Learn the law, not the solution" as the SciML position; a PINN baseline built and run |
