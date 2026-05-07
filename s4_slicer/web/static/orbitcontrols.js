/* Minimal Orbit-style controls for three.js (compatible with r149 global build).
 * Provides THREE.OrbitControls(camera, domElement) with .update(), .target,
 * .enableDamping. Supports rotate (LMB drag), pan (RMB or shift+LMB drag),
 * zoom (mouse wheel + pinch). Touch: 1-finger rotate, 2-finger pan/zoom.
 *
 * This is intentionally compact — it's not the full official OrbitControls
 * but covers the features the S4 web UI needs.
 */
(function () {
  if (typeof THREE === "undefined") {
    console.error("OrbitControls requires THREE to be loaded first.");
    return;
  }

  function OrbitControls(camera, dom) {
    this.object = camera;
    this.domElement = dom || document;
    this.target = new THREE.Vector3();
    this.enableDamping = false;
    this.dampingFactor = 0.08;
    this.rotateSpeed = 0.9;
    this.zoomSpeed   = 0.9;
    this.panSpeed    = 0.9;
    this.minDistance = 0.01;
    this.maxDistance = 1e7;
    this.minPolarAngle = 0;
    this.maxPolarAngle = Math.PI;

    const scope = this;

    const STATE = { NONE: 0, ROTATE: 1, PAN: 2, TOUCH_ROTATE: 3, TOUCH_PAN_ZOOM: 4 };
    let state = STATE.NONE;

    // spherical relative to target
    const spherical      = new THREE.Spherical();
    const sphericalDelta = new THREE.Spherical();
    let scale = 1;
    const panOffset = new THREE.Vector3();

    const rotateStart = new THREE.Vector2();
    const rotateEnd   = new THREE.Vector2();
    const rotateDelta = new THREE.Vector2();

    const panStart = new THREE.Vector2();
    const panEnd   = new THREE.Vector2();
    const panDelta = new THREE.Vector2();

    const offset = new THREE.Vector3();
    const quat   = new THREE.Quaternion().setFromUnitVectors(camera.up, new THREE.Vector3(0, 1, 0));
    const quatInv= quat.clone().invert();

    function getZoomScale() { return Math.pow(0.95, scope.zoomSpeed); }

    function rotateLeft(angle) { sphericalDelta.theta -= angle; }
    function rotateUp(angle)   { sphericalDelta.phi   -= angle; }

    const v = new THREE.Vector3();
    function panLeft(distance, m) {
      v.setFromMatrixColumn(m, 0);
      v.multiplyScalar(-distance);
      panOffset.add(v);
    }
    function panUp(distance, m) {
      v.setFromMatrixColumn(m, 1);
      v.multiplyScalar(distance);
      panOffset.add(v);
    }
    function pan(dx, dy) {
      const el = (scope.domElement === document) ? document.body : scope.domElement;
      if (camera.isPerspectiveCamera) {
        const pos = camera.position;
        const off = pos.clone().sub(scope.target);
        let dist = off.length();
        dist *= Math.tan((camera.fov / 2) * Math.PI / 180);
        panLeft(2 * dx * dist / el.clientHeight, camera.matrix);
        panUp(2 * dy * dist / el.clientHeight, camera.matrix);
      }
    }

    this.update = function () {
      offset.copy(camera.position).sub(scope.target);
      offset.applyQuaternion(quat);
      spherical.setFromVector3(offset);
      spherical.theta += sphericalDelta.theta;
      spherical.phi   += sphericalDelta.phi;
      spherical.phi = Math.max(scope.minPolarAngle, Math.min(scope.maxPolarAngle, spherical.phi));
      spherical.makeSafe();
      spherical.radius *= scale;
      spherical.radius = Math.max(scope.minDistance, Math.min(scope.maxDistance, spherical.radius));
      scope.target.add(panOffset);
      offset.setFromSpherical(spherical);
      offset.applyQuaternion(quatInv);
      camera.position.copy(scope.target).add(offset);
      camera.lookAt(scope.target);
      if (scope.enableDamping) {
        sphericalDelta.theta *= (1 - scope.dampingFactor);
        sphericalDelta.phi   *= (1 - scope.dampingFactor);
        panOffset.multiplyScalar(1 - scope.dampingFactor);
      } else {
        sphericalDelta.set(0, 0, 0);
        panOffset.set(0, 0, 0);
      }
      scale = 1;
    };

    function onPointerDown(e) {
      const isPan = e.button === 2 || e.shiftKey;
      if (isPan) {
        state = STATE.PAN;
        panStart.set(e.clientX, e.clientY);
      } else {
        state = STATE.ROTATE;
        rotateStart.set(e.clientX, e.clientY);
      }
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup",   onPointerUp);
      e.preventDefault();
    }
    function onPointerMove(e) {
      const el = (scope.domElement === document) ? document.body : scope.domElement;
      if (state === STATE.ROTATE) {
        rotateEnd.set(e.clientX, e.clientY);
        rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
        rotateLeft(2 * Math.PI * rotateDelta.x / el.clientHeight);
        rotateUp  (2 * Math.PI * rotateDelta.y / el.clientHeight);
        rotateStart.copy(rotateEnd);
      } else if (state === STATE.PAN) {
        panEnd.set(e.clientX, e.clientY);
        panDelta.subVectors(panEnd, panStart).multiplyScalar(scope.panSpeed);
        pan(panDelta.x, panDelta.y);
        panStart.copy(panEnd);
      }
    }
    function onPointerUp() {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup",   onPointerUp);
      state = STATE.NONE;
    }

    function onWheel(e) {
      e.preventDefault();
      if (e.deltaY < 0) scale /= getZoomScale();
      else              scale *= getZoomScale();
    }
    function onContextMenu(e) { e.preventDefault(); }

    // Touch
    let touchPrev = null;
    function onTouchStart(e) {
      if (e.touches.length === 1) {
        state = STATE.TOUCH_ROTATE;
        rotateStart.set(e.touches[0].clientX, e.touches[0].clientY);
      } else if (e.touches.length === 2) {
        state = STATE.TOUCH_PAN_ZOOM;
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        touchPrev = { dist: Math.hypot(dx, dy),
                      cx: (e.touches[0].clientX + e.touches[1].clientX) / 2,
                      cy: (e.touches[0].clientY + e.touches[1].clientY) / 2 };
      }
    }
    function onTouchMove(e) {
      e.preventDefault();
      const el = (scope.domElement === document) ? document.body : scope.domElement;
      if (state === STATE.TOUCH_ROTATE && e.touches.length === 1) {
        rotateEnd.set(e.touches[0].clientX, e.touches[0].clientY);
        rotateDelta.subVectors(rotateEnd, rotateStart).multiplyScalar(scope.rotateSpeed);
        rotateLeft(2 * Math.PI * rotateDelta.x / el.clientHeight);
        rotateUp  (2 * Math.PI * rotateDelta.y / el.clientHeight);
        rotateStart.copy(rotateEnd);
      } else if (state === STATE.TOUCH_PAN_ZOOM && e.touches.length === 2) {
        const dx = e.touches[0].clientX - e.touches[1].clientX;
        const dy = e.touches[0].clientY - e.touches[1].clientY;
        const dist = Math.hypot(dx, dy);
        const cx = (e.touches[0].clientX + e.touches[1].clientX) / 2;
        const cy = (e.touches[0].clientY + e.touches[1].clientY) / 2;
        if (touchPrev) {
          if (dist > touchPrev.dist) scale /= getZoomScale();
          else                       scale *= getZoomScale();
          pan(cx - touchPrev.cx, cy - touchPrev.cy);
        }
        touchPrev = { dist, cx, cy };
      }
    }
    function onTouchEnd() { state = STATE.NONE; touchPrev = null; }

    const el = scope.domElement;
    el.addEventListener("pointerdown",   onPointerDown);
    el.addEventListener("wheel",         onWheel, { passive: false });
    el.addEventListener("contextmenu",   onContextMenu);
    el.addEventListener("touchstart",    onTouchStart, { passive: true });
    el.addEventListener("touchmove",     onTouchMove, { passive: false });
    el.addEventListener("touchend",      onTouchEnd);

    this.dispose = function () {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("contextmenu", onContextMenu);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("touchend", onTouchEnd);
    };

    this.update();
  }

  THREE.OrbitControls = OrbitControls;
})();
