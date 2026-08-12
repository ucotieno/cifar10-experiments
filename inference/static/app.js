const fileInput = document.getElementById("file-input");
const previewContainer = document.getElementById("preview-container");
const preview = document.getElementById("preview");
const predictButton = document.getElementById("predict-button");
const status = document.getElementById("status");
const result = document.getElementById("result");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");

let selectedFile = null;

function selectImage(file) {
    if (!file || !file.type.startsWith("image/")) {
        status.textContent = "Please select an image.";
        return;
    }

    selectedFile = file;

    const imageUrl = URL.createObjectURL(file);

    preview.src = imageUrl;
    previewContainer.classList.remove("hidden");

    predictButton.disabled = false;

    result.classList.add("hidden");
    status.textContent = "";
}

fileInput.addEventListener("change", () => {
    selectImage(fileInput.files[0]);
});

predictButton.addEventListener("click", async () => {
    if (!selectedFile) {
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    predictButton.disabled = true;
    status.textContent = "Running inference...";
    result.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Inference failed.");
        }

        prediction.textContent = data.class;
        confidence.textContent =
            `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

        result.classList.remove("hidden");
        status.textContent = "";
    } catch (error) {
        status.textContent = error.message;
    } finally {
        predictButton.disabled = false;
    }
});
