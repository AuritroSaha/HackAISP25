function autofillCarDetails() {
    let model = document.getElementById("model").value;
    let year = document.getElementById("Model Year").value;
    let selectedCar = carData.find(car => car.Name === model && car["Model Year"] === year);

    if (selectedCar) {
        Object.keys(selectedCar).forEach(key => {
            let safeKey = key.replace(/[^a-zA-Z0-9]/g, '');
            let input = document.getElementById(safeKey) || document.getElementById(key);
            if (input) {
                input.value = selectedCar[key];
            }
        });
    }
}
