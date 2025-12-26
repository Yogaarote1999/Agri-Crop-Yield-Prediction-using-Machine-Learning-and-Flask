document.addEventListener('DOMContentLoaded', function() {
    // ---------------- Mobile menu toggle ----------------
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });

        document.addEventListener('click', function(event) {
            if (!hamburger.contains(event.target) && !navMenu.contains(event.target)) {
                navMenu.classList.remove('active');
                hamburger.classList.remove('active');
            }
        });
    }

    // ---------------- Hide/Show password toggle ----------------
    const togglePasswordButtons = document.querySelectorAll('.toggle-password');
    togglePasswordButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const input = this.previousElementSibling;
            const icon = this.querySelector('i');
            if (!input) return;
            if (input.type === 'password') {
                input.type = 'text';
                if (icon) { icon.classList.remove('fa-eye'); icon.classList.add('fa-eye-slash'); }
            } else {
                input.type = 'password';
                if (icon) { icon.classList.remove('fa-eye-slash'); icon.classList.add('fa-eye'); }
            }
        });
    });

    // ---------------- Prediction form logic ----------------
    const form = document.getElementById('predictForm');
    const resultEl = document.getElementById('result');

    if (form && resultEl) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const fd = new FormData(form);

            // IMPORTANT: Map HTML input names → Backend expected keys
            const data = {
                N: fd.get("N"),
                P: fd.get("P"),
                K: fd.get("K"),
                temperature: fd.get("temperature"),
                humidity: fd.get("humidity"),
                ph: fd.get("ph"),
                rainfall: fd.get("rainfall"),

                // 🔥 FIXED: Mapping training column names → API keys
                fertilizer: fd.get("Fertilizer_Usage_kg_per_hectare"),
                pesticide: fd.get("Pesticide_Usage_litre_per_hectare"),
                seed: fd.get("Seed_Expense_per_hectare(INR)"),
                other: fd.get("Other_Expense(INR)")
                market_price: fd.get("market_price")
            };

            resultEl.textContent = "⏳ Predicting... please wait...";

            try {
                const response = await fetch("/api/predict_all", {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                if (!response.ok) throw new Error('Server error');

                const result = await response.json();

                const predictedYield = parseFloat(result.Predicted_Yield.replace(/[^\d.]/g, '')).toFixed(2);
                const totalExpense = parseFloat(result.Total_Expense.replace(/[₹,]/g, '')).toFixed(2);
                const predictedRevenue = parseFloat(result.Predicted_Revenue.replace(/[₹,]/g, '')).toFixed(2);

                const netProfit = predictedRevenue - totalExpense;

                const profitOrLossText = netProfit >= 0
                    ? `📈 Profit: ₹${netProfit.toFixed(2)} Per Hectare`
                    : `📉 Loss: ₹${Math.abs(netProfit).toFixed(2)} Per Hectare`;

                resultEl.textContent = `
🌾 Predicted Crop: ${result.Predicted_Crop}
🌱 Predicted Yield: ${predictedYield} Kg/Per Hectare (${(predictedYield/1000).toFixed(2)} Tons/Per Hectare)
💰 Total Expense: ₹${totalExpense} Per Hectare
💵 Predicted Revenue: ₹${predictedRevenue} Per Hectare
${profitOrLossText}
                `;

            } catch (err) {
                resultEl.textContent = '❌ Error: ' + err.message;
            }
        });
    }
});
