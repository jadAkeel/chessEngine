const FILES = {
  move: "/sounds/move.mp3",
  capture: "/sounds/capture.mp3",
  check: "/sounds/check.mp3",
};

let enabled = true;
let context = null;
let loading = null;
const buffers = {};
const fallback = {};

export function setSoundEnabled(value) {
  enabled = Boolean(value);
}

function audioContext() {
  if (context) return context;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  context = new Ctor();
  return context;
}

function loadBuffers() {
  if (loading) return loading;
  const ctx = audioContext();
  if (!ctx) return Promise.resolve();
  loading = Promise.all(
    Object.entries(FILES).map(async ([key, url]) => {
      try {
        const data = await (await fetch(url)).arrayBuffer();
        // Callback form: older Safari has no promise-returning decodeAudioData.
        buffers[key] = await new Promise((resolve, reject) => ctx.decodeAudioData(data, resolve, reject));
      } catch (err) {
        console.warn("Could not load sound", key, err);
      }
    }),
  );
  return loading;
}

// Phones only let a page make sound after a touch, and only from inside that touch.
// The engine's reply arrives seconds later, so the audio context is unlocked on every
// gesture (iOS suspends it again when the tab goes to the background).
function unlock() {
  const ctx = audioContext();
  if (!ctx) return;
  if (ctx.state !== "running") {
    ctx.resume().catch(() => {});
    const silent = ctx.createBufferSource();
    silent.buffer = ctx.createBuffer(1, 1, 22050);
    silent.connect(ctx.destination);
    silent.start(0);
  }
  void loadBuffers();
}

let listening = false;
export function initSound() {
  if (listening || typeof window === "undefined") return;
  listening = true;
  for (const type of ["pointerdown", "touchend", "keydown"]) {
    window.addEventListener(type, unlock, { capture: true, passive: true });
  }
}

function play(key) {
  if (!enabled) return;
  const ctx = context;
  const buffer = buffers[key];
  if (ctx && buffer) {
    if (ctx.state !== "running") ctx.resume().catch(() => {});
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.start(0);
    return;
  }
  let audio = fallback[key];
  if (!audio) {
    audio = new Audio(FILES[key]);
    audio.preload = "auto";
    fallback[key] = audio;
  }
  audio.currentTime = 0;
  audio.play().catch(() => {});
}

export function playMoveSound() {
  play("move");
}

export function playCaptureSound() {
  play("capture");
}

export function playCheckSound() {
  play("check");
}

export function playMoveSoundFor(move, game) {
  if (!enabled) return;
  play(move.captured ? "capture" : "move");
  setTimeout(() => {
    if (game.isCheck()) play("check");
  }, 80);
}
