document.getElementById('prediction-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const latitude = parseFloat(document.getElementById('latitude-input').value);
    const longitude = parseFloat(document.getElementById('longitude-input').value);
    const depth = parseFloat(document.getElementById('depth-input').value);

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ latitude, longitude, depth })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        const resultDiv = document.getElementById('prediction-result');
        const category = result.predicted_category;

        // Determine the color based on the magnitude category
        let categoryColor = '';
        if (category === 'Low') {
            categoryColor = 'green';
        } else if (category === 'Medium') {
            categoryColor = 'yellow';
        } else if (category === 'High') {
            categoryColor = 'red';
        }

        // Construct the result HTML
        let resultHTML = `
            <p>Predicted Magnitude Category: <span style="color: ${categoryColor}; font-weight: bold;">${category}</span></p>
        `;

        // Only display the exact magnitude if the category is not 'Low'
        if (category !== 'Low') {
            resultHTML += `<p>Predicted Exact Magnitude: ${result.predicted_magnitude}</p>`;
        }

        resultDiv.innerHTML = resultHTML;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('prediction-result').innerText = `Error: ${error.message}`;
    }
});
