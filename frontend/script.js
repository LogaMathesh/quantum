async function predict() {

    const fertilizer =
        parseFloat(document.getElementById("fertilizer").value);

    const temp =
        parseFloat(document.getElementById("temp").value);

    const N =
        parseFloat(document.getElementById("N").value);

    const P =
        parseFloat(document.getElementById("P").value);

    const K =
        parseFloat(document.getElementById("K").value);


    // Check inputs

    if (
        isNaN(fertilizer) ||
        isNaN(temp) ||
        isNaN(N) ||
        isNaN(P) ||
        isNaN(K)
    ) {
        alert("Please enter all values.");
        return;
    }


    document.getElementById("loading").innerText =
        "Predicting...";


    try {

        const response = await fetch(
            "http://localhost:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    fertilizer: fertilizer,
                    temp: temp,
                    N: N,
                    P: P,
                    K: K
                })
            }
        );


        const result = await response.json();


        document.getElementById("rfResult").innerText =
            result.random_forest.toFixed(3);


        document.getElementById("quantumResult").innerText =
            result.quantum.toFixed(3);


        document.getElementById("differenceResult").innerText =
            result.difference.toFixed(3);


        document.getElementById("percentageResult").innerText =
            result.difference_percent.toFixed(2) + "%";


        document.getElementById("loading").innerText =
            "Prediction completed.";

    }

    catch (error) {

        console.error(error);

        document.getElementById("loading").innerText =
            "Could not connect to backend.";

    }
}