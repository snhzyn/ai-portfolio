function setupSegmentedControl(groupId, hiddenInputId) {
  const group = document.getElementById(groupId);
  const hiddenInput = document.getElementById(hiddenInputId);

  group.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      group.querySelectorAll(".segment").forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");
      hiddenInput.value = button.dataset.value;
    });
  });
}

function setStatus(status, text) {
  const badge = document.getElementById("status-badge");
  badge.className = `status-badge ${status}`;
  badge.textContent = text;
}

function showError(message) {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  const errorBox = document.getElementById("error-box");
  errorBox.textContent = "";
  errorBox.classList.add("hidden");
}

function renderList(elementId, items) {
  const list = document.getElementById(elementId);
  list.innerHTML = "";

  (items || []).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  });
}

function renderStoryboard(scenes) {
  const container = document.getElementById("storyboard-list");
  container.innerHTML = "";

  (scenes || []).forEach((scene) => {
    const block = document.createElement("div");
    block.className = "storyboard-scene";

    block.innerHTML = `
      <div class="storyboard-scene-title">Scene ${scene.scene}</div>
      <div class="storyboard-meta">${scene.time_range}</div>
      <p><strong>Visual:</strong> ${scene.visual}</p>
      <p><strong>Voiceover:</strong> ${scene.voiceover}</p>
      <p><strong>On-screen text:</strong> ${scene.on_screen_text}</p>
    `;

    container.appendChild(block);
  });
}

function renderResult(payload) {
  const result = payload.result || {};

  document.getElementById("empty-state").classList.add("hidden");
  document.getElementById("result-container").classList.remove("hidden");

  document.getElementById("final-topic").textContent =
    result.final_topic_suggestion ||
    result.editor_brief?.topic ||
    result.creative_brief?.normalized_topic ||
    "-";

  document.getElementById("hook").textContent =
    result.revised_script?.hook || result.best_script?.hook || "-";

  document.getElementById("revised-script").textContent =
    result.revised_script?.script || "-";

  renderList("titles-list", result.publish_package?.titles || []);
  renderList("thumbnail-list", result.publish_package?.thumbnail_text || []);
  renderStoryboard(result.storyboard_package?.scenes || []);

  document.getElementById("video-prompt").textContent =
    result.video_generation_prompt || "-";
}

document.addEventListener("DOMContentLoaded", () => {
  setupSegmentedControl("platform-group", "platform");
  setupSegmentedControl("language-group", "language");
  setupSegmentedControl("duration-group", "duration_sec");

  const form = document.getElementById("content-form");
  const button = document.getElementById("generate-btn");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    clearError();
    setStatus("loading", "Generating...");
    button.disabled = true;

    const body = {
      topic: document.getElementById("topic").value.trim(),
      platform: document.getElementById("platform").value,
      audience: document.getElementById("audience").value.trim(),
      tone: document.getElementById("tone").value.trim(),
      duration_sec: Number(document.getElementById("duration_sec").value),
      reference_text: document.getElementById("reference_text").value.trim(),
      language: document.getElementById("language").value,
    };

    try {
      const response = await fetch("/api/content/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to generate content.");
      }

      renderResult(data);
      setStatus("success", "Done");
    } catch (error) {
      showError(error.message || "Something went wrong.");
      setStatus("error", "Failed");
    } finally {
      button.disabled = false;
    }
  });
});