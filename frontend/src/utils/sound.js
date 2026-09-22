const sounds = {};

function preload(key, path) {
  const audio = new Audio(path);
  audio.preload = "auto";
  sounds[key] = audio;
}

let loaded = false;
function ensureLoaded() {
  if (loaded) return;
  preload("move", "/sounds/move.mp3");
  preload("capture", "/sounds/capture.mp3");
  preload("check", "/sounds/check.mp3");
  loaded = true;
}

export function playMoveSound() {
  ensureLoaded();
  sounds.move.currentTime = 0;
  sounds.move.play().catch(() => {});
}

export function playCaptureSound() {
  ensureLoaded();
  sounds.capture.currentTime = 0;
  sounds.capture.play().catch(() => {});
}

export function playCheckSound() {
  ensureLoaded();
  sounds.check.currentTime = 0;
  sounds.check.play().catch(() => {});
}

export function playMoveSoundFor(move, game) {
  if (move.captured) {
    playCaptureSound();
  } else {
    playMoveSound();
  }
  setTimeout(() => {
    if (game.isCheck()) {
      playCheckSound();
    }
  }, 80);
}
