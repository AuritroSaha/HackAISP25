function parseCSV(text) {
    const rows = text.trim().split("\n").map(row => row.split(","));
    const headers = rows.shift().map(h => h.trim());

    carData = rows.map(row => {
        let obj = {};
        headers.forEach((header, index) => {
            obj[header] = row[index] ? row[index].trim() : "";
        });
        return obj;
    });

    populateModelDropdown();
}

function populateModelDropdown() {
    let modelDropdown = document.getElementById("model");
    let uniqueModels = [...new Set(carData.map(car => car.Model).filter(Boolean))];
    modelDropdown.innerHTML = `<option value="">Select Model</option>` +
        uniqueModels.map(model => `<option value="${model}">${model}</option>`).join("");
}

