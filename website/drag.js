function autofillCarDetails() {
    let model = document.getElementById("model").value;
    let year = document.getElementById("year").value;
    let selectedCar = carData.find(car => car.Name === model && car.Year === year);

    if (selectedCar) {
        Object.keys(selectedCar).forEach(key => {
            const id = key.replace(/\s+/g, '_').replace(/[^\w]/g, '');
            if(document.getElementById(id)){
                document.getElementById(id).value = selectedCar[key];
            }
        });
    }
}
