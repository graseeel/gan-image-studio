const artifacts = {
  training: {
    src: "assets/smoke-training-grid.png",
    alt: "A local CPU smoke-test training sample grid from GAN Image Studio.",
    caption: "First sample grid emitted during the smoke training path.",
    command: "gan-studio train --dataset folder --max-batches 2",
    proof: "Sampling, grid writing, and local artifact paths work.",
  },
  seed: {
    src: "assets/smoke-seed-228.png",
    alt: "A deterministic seed 228 generation grid from a smoke-test checkpoint.",
    caption: "Seed 228 generated from the smoke-test checkpoint.",
    command: "gan-studio generate checkpoint.pt --seed 228 --count 16",
    proof: "Checkpoint loading, deterministic latent sampling, and grid export work.",
  },
};

const viewer = document.querySelector(".artifact-viewer");
const image = document.querySelector("#artifact-image");
const caption = document.querySelector("#artifact-caption");
const command = document.querySelector("#artifact-command");
const proof = document.querySelector("#artifact-proof");
const tabs = Array.from(document.querySelectorAll(".artifact-tab"));
const copyStatus = document.querySelector("#copy-status");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let swapTimer = 0;

function setArtifact(key) {
  const artifact = artifacts[key];
  if (!artifact || !viewer || !image || !caption || !command || !proof) return;
  if (tabs.find((tab) => tab.dataset.artifact === key)?.getAttribute("aria-pressed") === "true") {
    return;
  }

  tabs.forEach((tab) => {
    tab.setAttribute("aria-pressed", String(tab.dataset.artifact === key));
  });

  const commit = () => {
    image.src = artifact.src;
    image.alt = artifact.alt;
    caption.textContent = artifact.caption;
    command.textContent = artifact.command;
    proof.textContent = artifact.proof;
  };

  if (reduceMotion.matches) {
    commit();
    return;
  }

  window.clearTimeout(swapTimer);
  viewer.dataset.swapping = "true";
  swapTimer = window.setTimeout(() => {
    commit();
    viewer.dataset.swapping = "false";
  }, 120);
}

async function copyCommand(text, button) {
  if (!text || !copyStatus) return;

  try {
    await navigator.clipboard.writeText(text);
    copyStatus.textContent = "Command copied.";
    const previous = button.textContent;
    button.textContent = "Copied";
    window.setTimeout(() => {
      button.textContent = previous;
      copyStatus.textContent = "";
    }, 1200);
  } catch {
    copyStatus.textContent = "Copy failed. Select the command text instead.";
  }
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setArtifact(tab.dataset.artifact);
  });
});

document.querySelectorAll(".copy-button").forEach((button) => {
  button.addEventListener("click", () => {
    copyCommand(button.dataset.copy, button);
  });
});
