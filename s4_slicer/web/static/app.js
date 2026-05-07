/* S4 Unified Slicer — Web UI client.
 *
 * Handles upload, parameter form, SSE progress stream, log terminal,
 * and 3D previews (input STL + final 4-axis path) via three.js.
 */

(() => {
  const $ = (id) => document.getElementById(id);

  const state = {
    job_id: null,
    inputPreview: null,
    polar: true,
    eventSrc: null,
  };

  window.onerror = (msg, url, line) => {
    addLog(`[ui] JS Error: ${msg} (line ${line})`, "error");
  };

  // ---------- Upload ------------------------------------------------------
  const dropzone = $("dropzone");
  const stlInput = $("stl-input");
  const fileInfo = $("file-info");

  dropzone.addEventListener("click", () => stlInput.click());
  ["dragover", "dragenter"].forEach(ev =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(ev =>
    dropzone.addEventListener(ev, () => dropzone.classList.remove("drag")));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) {
      stlInput.files = e.dataTransfer.files;
      handleFile(e.dataTransfer.files[0]);
    }
  });
  stlInput.addEventListener("change", () => {
    if (stlInput.files.length) handleFile(stlInput.files[0]);
  });

  async function handleFile(file) {
    fileInfo.textContent = `${file.name} — ${(file.size / 1024).toFixed(1)} KB`;
    addLog(`[ui] uploading ${file.name} (${file.size} bytes)`, "info");
    const fd = new FormData();
    fd.append("file", file);
    let resp;
    try {
      resp = await fetch("/api/upload", { method: "POST", body: fd });
    } catch (e) {
      addLog(`[ui] upload failed: ${e}`, "error");
      return;
    }
    if (!resp.ok) {
      addLog(`[ui] upload failed: HTTP ${resp.status}`, "error");
      return;
    }
    const j = await resp.json();
    state.job_id = j.job_id;
    state.inputPreview = j.preview;
    addLog(`[ui] uploaded — job_id=${j.job_id}`, "ok");
    try {
      renderInput(j.preview);
    } catch (e) {
      addLog(`[ui] preview render failed: ${e}`, "error");
    }
    $("start-btn").disabled = false;
    $("stop-btn").disabled  = false;
    $("progress-label").textContent = `Ready — ${file.name}`;
  }

  // ---------- Slice button ------------------------------------------------
  $("start-btn").addEventListener("click", startSlice);
  $("stop-btn").addEventListener("click", () => location.reload());

  function gatherParams() {
    return {
      max_overhang: +$("p-max-overhang").value,
      rot_iter:     +$("p-rot-iter").value,
      deform_iter:  +$("p-deform-iter").value,
      num_passes:   +$("p-num-passes").value,
      layer_height: +$("p-layer-height").value,
      perimeters:   +$("p-perimeters").value,
      fill_density: $("p-fill-density").value,
      seg_size:     +$("p-seg-size").value,
      cartesian:    $("p-cartesian").checked,
      fast:         $("p-fast").checked,
    };
  }

  async function startSlice() {
    if (!state.job_id) return;
    state.polar = !$("p-cartesian").checked;
    $("backend-tag").textContent = $("p-fast").checked ? "fast backend" : "reference backend";
    $("start-btn").disabled = true;
    $("term-status").textContent = "running";
    setProgress(0, "starting…");
    clearStages();
    clearTerminal();
    clearResults();
    addLog("[ui] starting pipeline…", "info");

    const params = gatherParams();
    const resp = await fetch(`/api/start/${state.job_id}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body:    JSON.stringify(params),
    });
    if (!resp.ok) {
      const t = await resp.text();
      addLog(`[ui] start failed: HTTP ${resp.status} ${t}`, "error");
      $("start-btn").disabled = false;
      return;
    }
    openStream(state.job_id);
  }

  // ---------- SSE stream --------------------------------------------------
  function openStream(jobId) {
    if (state.eventSrc) state.eventSrc.close();
    const src = new EventSource(`/api/stream/${jobId}`);
    state.eventSrc = src;

    src.onmessage = (msg) => {
      let evt;
      try { evt = JSON.parse(msg.data); } catch { return; }
      handleEvent(evt);
    };
    src.addEventListener("hello", (m) => addLog("[sse] connected", "info"));
    src.onerror = () => {
      addLog("[sse] connection closed", "warn");
      src.close();
    };
  }

  function handleEvent(evt) {
    if (evt.type === "log") {
      addLog(evt.message, evt.level || "info", evt.ts);
    } else if (evt.type === "progress") {
      const pct = (evt.global_progress || 0) * 100;
      setProgress(pct, evt.stage_label || "");
      markStage(evt.stage, evt.stage_progress || 0);
    } else if (evt.type === "error") {
      $("progress-bar").classList.add("error");
      setProgress(100, "ERROR");
      addLog(`✗ ${evt.message}`, "error");
      $("term-status").textContent = "failed";
      addResultPill(`failed: ${evt.message}`, "err");
      $("start-btn").disabled = false;
    } else if (evt.type === "done") {
      $("progress-bar").classList.add("done");
      setProgress(100, "Done");
      addLog(`✓ pipeline complete in ${(+evt.elapsed).toFixed(1)}s`, "ok");
      $("term-status").textContent = `done in ${(+evt.elapsed).toFixed(1)}s`;
      addResultPill(`elapsed: ${(+evt.elapsed).toFixed(1)}s`, "ok");
      if (evt.stats && evt.stats.output_lines) {
        addResultPill(`${evt.stats.output_lines.toLocaleString()} g-code lines`, "ok");
      }
      addDownloadButtons(state.job_id);
      $("start-btn").disabled = false;
      // Fetch path preview & render.
      fetch(`/api/preview/${state.job_id}/path`).then(r => r.json()).then(renderPath)
          .catch(e => addLog(`[ui] path preview fetch failed: ${e}`, "warn"));
    }
  }

  // ---------- Progress / stages -----------------------------------------
  const STAGES = ["load", "tet", "rotation", "deform", "slice", "transform"];
  const STAGE_NAMES = {
    load: "load", tet: "tetrahedralise", rotation: "rotation",
    deform: "deform", slice: "slice", transform: "transform",
  };

  function clearStages() {
    const el = $("stages");
    el.innerHTML = "";
    STAGES.forEach((s) => {
      const d = document.createElement("div");
      d.className = "stage";
      d.dataset.key = s;
      d.textContent = STAGE_NAMES[s];
      el.appendChild(d);
    });
    $("progress-bar").classList.remove("error", "done");
  }

  function markStage(key, prog) {
    let foundActive = false;
    document.querySelectorAll(".stage").forEach((el) => {
      const k = el.dataset.key;
      if (k === key) {
        el.classList.add("active"); el.classList.remove("done");
        foundActive = true;
      } else if (foundActive) {
        el.classList.remove("active", "done");
      } else {
        el.classList.add("done"); el.classList.remove("active");
      }
    });
  }

  function setProgress(pct, label) {
    $("progress-bar").querySelector(".fill").style.width = pct.toFixed(1) + "%";
    $("progress-pct").textContent = pct.toFixed(0) + "%";
    if (label) $("progress-label").textContent = label;
  }

  // ---------- Terminal ----------------------------------------------------
  function clearTerminal() { $("terminal").innerHTML = ""; }
  function addLog(msg, level, ts) {
    const el = $("terminal");
    const line = document.createElement("div");
    line.className = level || "info";
    const tsStr = (ts != null) ? `[${(+ts).toFixed(1)}s]` : `[${(performance.now() / 1000).toFixed(1)}s]`;
    line.innerHTML = `<span class="ts">${tsStr}</span>${escapeHTML(msg)}`;
    el.appendChild(line);
    el.scrollTop = el.scrollHeight;
  }
  function escapeHTML(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

  // ---------- Results pills ----------------------------------------------
  function clearResults() { $("results").innerHTML = ""; }
  function addResultPill(text, kind) {
    const d = document.createElement("div");
    d.className = "pill " + (kind || "");
    d.textContent = text;
    $("results").appendChild(d);
  }
  function addDownloadButtons(jobId) {
    const r = $("results");
    const a1 = document.createElement("a");
    a1.href = `/api/download/${jobId}/gcode`;
    a1.className = "pill ok"; a1.style.textDecoration = "none";
    a1.textContent = "↓ download .gcode"; a1.target = "_blank";
    r.appendChild(a1);
    const a2 = document.createElement("a");
    a2.href = `/api/download/${jobId}/stl`;
    a2.className = "pill"; a2.style.textDecoration = "none";
    a2.textContent = "↓ deformed .stl"; a2.target = "_blank";
    r.appendChild(a2);
  }

  // ---------- 3D viewports ----------------------------------------------
  // Two scenes — one for input STL, one for path preview.
  const viewers = {};
  function makeViewer(canvas) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0d12);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(60, -90, 80);
    camera.up.set(0, 0, 1);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir = new THREE.DirectionalLight(0xffffff, 0.7);
    dir.position.set(50, -100, 100); scene.add(dir);
    const grid = new THREE.GridHelper(200, 20, 0x2a2f3a, 0x1c1f27);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    const ax = new THREE.AxesHelper(15); scene.add(ax);
    const controls = new THREE.OrbitControls(camera, canvas);
    controls.enableDamping = true;
    function resize() {
      const w = canvas.clientWidth; const h = canvas.clientHeight;
      if (w === 0 || h === 0) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h; camera.updateProjectionMatrix();
    }
    function loop() {
      resize(); controls.update(); renderer.render(scene, camera);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
    return { renderer, scene, camera, controls, content: new THREE.Group() };
  }
  viewers.input = makeViewer($("canvas-input"));
  viewers.path  = makeViewer($("canvas-path"));
  viewers.input.scene.add(viewers.input.content);
  viewers.path.scene.add(viewers.path.content);

  function fitCamera(viewer, box) {
    if (!box) return;
    const center = box.getCenter(new THREE.Vector3());
    const size   = box.getSize(new THREE.Vector3());
    const radius = size.length() * 0.6 + 1;
    viewer.controls.target.copy(center);
    const dir = new THREE.Vector3(0.7, -1, 0.7).normalize();
    viewer.camera.position.copy(center.clone().add(dir.multiplyScalar(radius * 1.6)));
    viewer.camera.near = Math.max(0.1, radius / 100);
    viewer.camera.far  = radius * 100;
    viewer.camera.updateProjectionMatrix();
  }

  function renderInput(prev) {
    const v = viewers.input;
    while (v.content.children.length) v.content.remove(v.content.children[0]);
    if (!prev || !prev.vertices || !prev.vertices.length) return;

    addLog(`[ui] rendering input mesh...`, "info");
    // Manual flatten for better compatibility
    const verts = new Float32Array(prev.vertices.length * 3);
    for (let i = 0; i < prev.vertices.length; i++) {
      verts[i*3] = prev.vertices[i][0];
      verts[i*3+1] = prev.vertices[i][1];
      verts[i*3+2] = prev.vertices[i][2];
    }
    const faces = new Uint32Array(prev.faces.length * 3);
    for (let i = 0; i < prev.faces.length; i++) {
      faces[i*3] = prev.faces[i][0];
      faces[i*3+1] = prev.faces[i][1];
      faces[i*3+2] = prev.faces[i][2];
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(faces, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshLambertMaterial({ color: 0x4fc3f7, flatShading: true,
      transparent: true, opacity: 0.85, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geo, mat);
    v.content.add(mesh);
    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x6cd2ff, transparent: true, opacity: 0.15 }));
    v.content.add(wire);
    fitCamera(v, new THREE.Box3().setFromObject(mesh));
    $("vp-input-ph").style.display = "none";
    $("vp-input-meta").textContent = `${prev.vertices.length.toLocaleString()} verts · ${prev.faces.length.toLocaleString()} faces`;
  }

  function renderPath(prev) {
    const v = viewers.path;
    while (v.content.children.length) v.content.remove(v.content.children[0]);
    const pts = prev.points;
    const rots = prev.rotation;
    if (!pts || pts.length < 2) {
      addLog("[ui] empty path preview", "warn");
      return;
    }
    const segE = []; const segT = [];
    const colE = []; 
    const c = new THREE.Color();

    for (let i = 1; i < pts.length; i++) {
      const a = pts[i-1]; const b = pts[i];
      const dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
      const d2 = dx*dx + dy*dy + dz*dz;
      if (d2 > 10000) continue;
      
      const dist = Math.sqrt(d2);

      if (prev.extruding[i] && prev.extruding[i-1]) {
        segE.push(...a, ...b);
        // Color based on segment length (0.0 to 5.0mm scale)
        const t = Math.min(1.0, dist / 5.0);
        c.setHSL(0.7 * (1.0 - t), 1.0, 0.5); // Blue (short) -> Red (long)
        colE.push(c.r, c.g, c.b, c.r, c.g, c.b);
      } else {
        segT.push(...a, ...b);
      }
    }

    const mkLine = (arr, colArr, baseCol, opacity, useColors) => {
      if (!arr.length) return null;
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arr), 3));
      if (useColors && colArr) {
        g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(colArr), 3));
      }
      const m = new THREE.LineBasicMaterial({ 
        color: useColors ? 0xffffff : baseCol, 
        vertexColors: useColors,
        transparent: true, 
        opacity 
      });
      return new THREE.LineSegments(g, m);
    };

    const lE = mkLine(segE, colE, 0x66ffa0, 0.95, true);
    if (lE) v.content.add(lE);
    
    const lT = mkLine(segT, null, 0xff8a65, 0.15, false);
    if (lT) v.content.add(lT);

    const box = new THREE.Box3().setFromObject(v.content);
    fitCamera(v, box);
    $("vp-path-ph").style.display = "none";
    $("vp-path-meta").textContent = `${prev.n_lines.toLocaleString()} lines · showing ${prev.downsampled_to.toLocaleString()}`;
  }

  // initial empty state
  clearStages();
})();
