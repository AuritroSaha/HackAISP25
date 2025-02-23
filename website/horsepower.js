let carData = [];

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

function populateYearDropdown() {
    let model = document.getElementById("model").value;
    let yearDropdown = document.getElementById("year");
    let years = [...new Set(carData.filter(car => car.Model === model).map(car => car.Year))];
    yearDropdown.innerHTML = `<option value="">Select Year</option>` +
        years.map(year => `<option value="${year}">${year}</option>`).join("");
}

function autofillCarDetails() {
    let model = document.getElementById("model").value;
    let year = document.getElementById("year").value;
    let selectedCar = carData.find(car => car.Model === model && car.Year === year);

    if (selectedCar) {
        Object.keys(selectedCar).forEach(key => {
            let elementId = key.replace(/\s+/g, '_');
            if(document.getElementById(elementId)){
                document.getElementById(elementId).value = selectedCar[key];
            }
        });
    }
}

function getPrediction() {
    document.getElementById("horsepower_pred").innerText = document.getElementById("Horsepower").value || "Not available";
}

fetch('used_cars_data.csv')
    .then(response => response.text())
    .then(text => parseCSV(text));
