// Interactive sketch of the notched backbone for the project page.
// Constant curvature only: each module bends by theta = dl / d. The real twin (RA-L/) also has
// axial compression, slack tendons and gravity.
(function () {
  const svg = document.getElementById("robot");
  if (!svg) return;
  const pull = document.getElementById("pull");
  const pullOut = document.getElementById("pull-out");
  const readout = document.getElementById("readout");
  const modBtns = [...document.querySelectorAll("[data-modules]")];
  const mountBtns = [...document.querySelectorAll("[data-mount]")];

  const NS = "http://www.w3.org/2000/svg";
  const BODY = 150, COUPLER = 10, PITCH = BODY + COUPLER; // mm
  const R = 12.5, D = 9.5;                                  // outer radius, tendon offset (mm)
  const ROWS = 12;                                          // notch rows per module, alternating planes
  const W = 640, H = 560;
  let state = { modules: 2, mount: "upright", pull: 0 };

  const el = (name, attrs, parent) => {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    (parent || svg).appendChild(e);
    return e;
  };

  function centreline(n, dl) {
    const theta = dl / D;                 // bend per module (rad)
    const kappa = theta / BODY;           // curvature inside a body
    const L = n * PITCH, ds = 0.5;
    const pts = [];
    let x = 0, y = 0, phi = 0;
    for (let s = 0; s <= L + 1e-9; s += ds) {
      pts.push({ s, x, y, phi });
      const inBody = (s % PITCH) < BODY;
      phi += (inBody ? kappa : 0) * ds;
      x += Math.sin(phi) * ds;
      y += Math.cos(phi) * ds;
    }
    return { pts, theta };
  }

  function radiusAt(s) {
    const local = s % PITCH;
    if (local >= BODY) return R;
    let r = R;
    for (let k = 0; k < ROWS; k += 2) {               // in-plane family: hourglass waists
      const c = 8 + (k + 0.5) * (BODY - 16) / ROWS;
      r -= 6.5 * Math.exp(-(((local - c) / 2.4) ** 2));
    }
    return r;
  }

  // Scale and base position that keep every pull from -12 to +12 mm on the sheet (computed once per layout)
  const fitCache = {};
  function fitFor(n, hang) {
    const key = n + (hang ? "h" : "u");
    if (fitCache[key]) return fitCache[key];
    let x0 = -R, x1 = R, y0 = 0, y1 = 0;
    for (let dl = 0; dl <= 12; dl += 0.5) {
      for (const p of centreline(n, dl).pts) {
        x0 = Math.min(x0, p.x - R, -p.x - R); x1 = Math.max(x1, p.x + R, -p.x + R);
        y0 = Math.min(y0, p.y - R); y1 = Math.max(y1, p.y + R);
      }
    }
    const padX = 40, padTop = 60, padBot = 50, base = 34;
    const scale = Math.min(1.45, (W - 2 * padX) / (x1 - x0), (H - padTop - padBot - base) / (y1 - y0));
    const bx = W / 2 - ((x0 + x1) / 2) * scale;
    // upright: y grows upwards from the base; hanging: downwards
    const by = hang ? padTop - 20 + base - y0 * scale : H - padBot - base + y0 * scale;
    return (fitCache[key] = { scale, bx, by });
  }

  function draw() {
    const { modules: n, mount, pull: dl } = state;
    svg.querySelectorAll(".dyn").forEach((e) => e.remove());
    const g = el("g", { class: "dyn" });
    const { pts, theta } = centreline(n, dl);
    const hang = mount === "hanging";
    const sgn = hang ? 1 : -1;
    const fit = fitFor(n, hang);
    const scale = fit.scale, bx = fit.bx, by = fit.by;
    const P = (p, off) => {
      const nx = Math.cos(p.phi), ny = -Math.sin(p.phi);
      return [bx + (p.x + off * nx) * scale, by + sgn * (p.y + off * ny) * scale];
    };

    // axis of the straight robot
    el("line", { x1: bx, y1: by, x2: bx, y2: by + sgn * n * PITCH * scale, class: "axis" }, g);

    // body outline per module (orange), couplers (blue)
    for (let m = 0; m < n; m++) {
      const seg = pts.filter((p) => p.s >= m * PITCH && p.s <= m * PITCH + BODY);
      const left = seg.map((p) => P(p, radiusAt(p.s)));
      const right = seg.map((p) => P(p, -radiusAt(p.s))).reverse();
      const d = "M" + left.concat(right).map((q) => q[0].toFixed(1) + "," + q[1].toFixed(1)).join("L") + "Z";
      el("path", { d, class: "body" }, g);
      for (let k = 1; k < ROWS; k += 2) {               // out-of-plane family: slots across the tube
        const c = m * PITCH + 8 + (k + 0.5) * (BODY - 16) / ROWS;
        const p = pts.reduce((a, b) => (Math.abs(b.s - c) < Math.abs(a.s - c) ? b : a));
        const a = P(p, R - 1.2), b = P(p, -R + 1.2);
        el("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: "slot" }, g);
      }
      const cp = pts.filter((p) => p.s >= m * PITCH + BODY && p.s <= (m + 1) * PITCH);
      if (cp.length > 1) {
        const l2 = cp.map((p) => P(p, R + 0.6)), r2 = cp.map((p) => P(p, -R - 0.6)).reverse();
        el("path", { d: "M" + l2.concat(r2).map((q) => q.join(",")).join("L") + "Z", class: "coupler" }, g);
      }
    }
    // tendon on the pulled side
    const side = dl >= 0 ? -1 : 1;
    const tend = pts.filter((_, i) => i % 4 === 0).map((p) => P(p, side * D));
    el("polyline", { points: tend.map((q) => q.join(",")).join(" "), class: "tendon" }, g);

    // base column
    const bw = 44, bh = 26;
    el("rect", { x: bx - bw / 2, y: hang ? by - bh : by, width: bw, height: bh, rx: 3, class: "base" }, g);
    el("rect", { x: bx - 70, y: hang ? by - bh - 6 : by + bh, width: 140, height: 6, rx: 2, class: "base" }, g);

    // tip and dimension line for the lateral deflection
    const tip = pts[pts.length - 1];
    const T = P(tip, 0);
    el("circle", { cx: T[0], cy: T[1], r: 4.5, class: "tip" }, g);
    if (Math.abs(tip.x) > 3) {
      const yDim = T[1] + (hang ? 22 : -22);
      el("line", { x1: bx, y1: yDim, x2: T[0], y2: yDim, class: "dim", "marker-start": "url(#arr)", "marker-end": "url(#arr)" }, g);
      el("line", { x1: T[0], y1: T[1], x2: T[0], y2: yDim, class: "ext" }, g);
      const t = el("text", { x: (bx + T[0]) / 2, y: yDim + (hang ? 16 : -7), class: "dimtxt", "text-anchor": "middle" }, g);
      t.textContent = Math.abs(tip.x).toFixed(0) + " mm";
    }
    const deg = (theta * 180 / Math.PI);
    readout.textContent = `${n} module${n > 1 ? "s" : ""}, ${n * PITCH} mm long. Each module bends ${Math.abs(deg).toFixed(0)}°; ` +
      `the tip moves ${Math.abs(tip.x).toFixed(0)} mm sideways.`;
    pullOut.textContent = (dl > 0 ? "+" : "") + dl.toFixed(1) + " mm";
  }

  let touched = false;
  pull.addEventListener("input", () => { touched = true; state.pull = parseFloat(pull.value); draw(); });
  const choose = (btns, key, attr) => btns.forEach((b) => b.addEventListener("click", () => {
    btns.forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
    touched = true;
    state[key] = key === "modules" ? parseInt(b.dataset[attr], 10) : b.dataset[attr];
    draw();
  }));
  choose(modBtns, "modules", "modules");
  choose(mountBtns, "mount", "mount");

  // one gentle bend on load, unless the reader prefers no motion
  const target = 5.0;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    state.pull = target; pull.value = target; draw();
  } else {
    draw();
    const t0 = performance.now(), dur = 1400;
    const step = (t) => {
      if (touched) return;
      const u = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - u, 3);
      state.pull = +(target * e).toFixed(2); pull.value = state.pull; draw();
      if (u < 1) requestAnimationFrame(step);
    };
    setTimeout(() => requestAnimationFrame(step), 350);
  }
})();
