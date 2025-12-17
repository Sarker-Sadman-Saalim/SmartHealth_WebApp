import pandas as pd

df = pd.read_csv("data/merged_obesity_clean.csv")

questions = [
    "What should I focus on first?",
    "Why did I get this obesity category?",
    "How can I reduce my obesity risk?",
    "What lifestyle change matters the most?",
    "Give me a simple action plan."
]

rows = []

for _, row in df.iterrows():
    bmi = row["Weight"] / (row["Height"] ** 2)
    faf = row["FAF"]
    ch2o = row["CH2O"]
    fcvc = row["FCVC"]

    # Determine factors
    factors = []
    if bmi >= 30:
        factors.append("High BMI")
    if faf <= 1:
        factors.append("Low physical activity")
    if ch2o < 2:
        factors.append("Low water intake")
    if row["family_history_with_overweight"] == "yes":
        factors.append("Family history")

    factors = (factors + ["", "", ""])[:3]

    prediction = "Obesity_Type_III" if bmi >= 30 else "Overweight"

    for q in questions:
        answer = (
            f"Your predicted category is {prediction} because your BMI is {bmi:.1f} "
            f"and factors like {factors[0].lower()} contribute to your risk. "
            f"The most important action to take first is improving physical activity, "
            f"such as walking 20–30 minutes per day. "
            f"Once this improves, focus on diet quality and hydration for better results.\n<END>"
        )

        rows.append({
            "question": q,
            "answer": answer
        })

qa_df = pd.DataFrame(rows)
qa_df.to_csv("data/synthetic_qa_data_v2.csv", index=False)

print(f"Generated {len(qa_df)} improved Q&A pairs.")
