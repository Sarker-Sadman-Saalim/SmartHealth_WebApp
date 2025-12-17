import pandas as pd

# Load your obesity dataset (replace with your actual dataset path)
df = pd.read_csv("data/merged_obesity.csv")

# Initialize lists for questions and answers
questions = []
answers = []

# Question patterns for various obesity-related topics (now including TUE, SCC, and CAEC)
question_templates = [
    "What is my BMI if my height is {height} meters and my weight is {weight} kg?",
    "How many days per week am I physically active if my FAF is {faf}?",
    "How does my family history of obesity affect my risk of obesity?",
    "How much water should I drink per day based on my hydration level (CH2O: {ch2o})?",
    "What are the risks of obesity for someone with a family history of overweight?",
    "How does my age impact my obesity risk?",
    "What is the recommended amount of physical activity to reduce obesity?",
    "How does my transportation method (MTRANS: {mtrans}) affect my obesity risk?",
    "Does smoking (SMOKE: {smoke}) affect my weight or obesity risk?",
    "What role does vegetable consumption (FCVC: {fcvc}) play in obesity management?",
    "How does my number of meals per day (NCP: {ncp}) impact my obesity risk?",
    "What is the impact of consuming high-calorie foods regularly (FAVC: {favc}) on obesity?",
    "How does my body mass index (BMI: {bmi}) affect my risk of obesity?",
    "Does my hydration level (CH2O: {ch2o}) help in reducing obesity?",
    "How can I improve my obesity risk if I have a BMI of {bmi}?",
    "How does the amount of time I spend using technology (TUE: {tue}) affect my obesity?",
    "How does monitoring my calorie intake (SCC: {scc}) affect my obesity risk?",
    "What is the impact of eating between meals (CAEC: {caec}) on my obesity risk?"
]

# Randomly generate answers based on template
def generate_answer(question, row):
    if "BMI" in question:
        return f"Your BMI is calculated as weight divided by height squared. Based on your data, your BMI is {row['Weight'] / (row['Height'] ** 2):.2f}."
    elif "FAF" in question:
        return f"Your physical activity level (FAF) is {row['FAF']} days per week. Increasing your FAF to at least 3 days a week is recommended for weight management."
    elif "family history" in question:
        return f"Family history increases your risk of obesity. You {'do' if row['family_history_with_overweight'] == 'yes' else 'do not'} have a family history of obesity."
    elif "hydration" in question:
        return f"You should aim for 2-3 liters of water per day. Based on your hydration level (CH2O), you're currently drinking {row['CH2O']} liters."
    elif "physical activity" in question:
        return f"Increasing your physical activity to 3-5 days per week can help with weight management and reduce obesity risk."
    elif "transportation" in question:
        return f"Using active transportation (like walking or cycling) reduces obesity risk. Based on your data, you use {row['MTRANS']} as your main transportation."
    elif "smoking" in question:
        return f"Smoking affects metabolism and appetite. It's important to focus on a balanced diet and regular exercise to manage obesity, even if you're a smoker."
    elif "vegetable consumption" in question:
        return f"Consuming vegetables regularly can help with obesity prevention. Your current vegetable consumption level (FCVC) is {row['FCVC']}."
    elif "meals per day" in question:
        return f"Reducing the number of meals per day might help in reducing calorie intake. You currently eat {row['NCP']} main meals per day."
    elif "high-calorie foods" in question:
        return f"Frequent consumption of high-calorie foods (FAVC) is linked to higher obesity risk. You consume high-calorie foods {'regularly' if row['FAVC'] == 'yes' else 'rarely or never'}."
    elif "obesity" in question:
        return f"Your BMI of {row['Weight'] / (row['Height'] ** 2):.2f} is an indicator of obesity risk. A BMI over 30 is considered obese."
    elif "improve" in question:
        return f"To improve your obesity risk, focus on increasing physical activity and improving your diet. Based on your BMI of {row['Weight'] / (row['Height'] ** 2):.2f}, lifestyle changes are recommended."
    elif "TUE" in question:
        return f"Spending too much time using technology (TUE) can reduce your physical activity and contribute to weight gain. Reducing screen time can help with obesity prevention."
    elif "SCC" in question:
        return f"Monitoring your calorie intake (SCC) helps to control weight. Being aware of your calorie consumption can help with weight management and obesity prevention."
    elif "CAEC" in question:
        return f"Eating between meals (CAEC) can contribute to higher calorie intake and obesity risk. It’s recommended to avoid frequent snacking and focus on balanced meals."
    
# Generate 1000 unique Q&A pairs
for _, row in df.iterrows():
    for template in question_templates:
        # Generate question using the template and fill it with data from the row
        question = template.format(
            height=row['Height'],
            weight=row['Weight'],
            faf=row['FAF'],
            ch2o=row['CH2O'],
            mtrans=row['MTRANS'],
            smoke=row['SMOKE'],
            fcvc=row['FCVC'],
            ncp=row['NCP'],
            favc=row['FAVC'],
            bmi=row['Weight'] / (row['Height'] ** 2),
            tue=row['TUE'],
            scc=row['SCC'],
            caec=row['CAEC']
        )
        
        # Generate the corresponding answer
        answer = generate_answer(question, row)
        
        # Append to the questions and answers list
        questions.append(question)
        answers.append(answer)

# Create a DataFrame with the generated questions and answers
qa_df = pd.DataFrame({"question": questions, "answer": answers})

# Save the dataset
qa_df.to_csv("data/synthetic_qa_data.csv", index=False)

print(f"Generated {len(qa_df)} Q&A pairs.")
