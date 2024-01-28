function impute() {
    document.getElementById("resultID").value = "";
    var xhr = new XMLHttpRequest();

    xhr.onreadystatechange = function () {
        if (xhr.readyState == 4 && xhr.status == 200) {
            var result = String.fromCharCode.apply(null, new Uint8Array(this.response));
            document.getElementById("resultID").value = result;
        }
    }
    xhr.responseType = "arraybuffer";

    xhr.open("POST", "https://bbaeop16fundn0oqj8vr.containers.yandexcloud.net/predict_yfull");
    xhr.setRequestHeader("Content-Type", "text/plain");

    xhr.send(document.getElementById("sampleID").value);
}