import json

qa_pairs = [
    {"user": "Who created the heavens and the earth?", "ai": "In the beginning, God created the heavens and the earth."},
    {"user": "Explain Matthew 10:1", "ai": "In Matthew 10:1, Jesus called his twelve disciples to him and gave them authority to drive out impure spirits and to heal every disease and sickness."},
    {"user": "Who was the first man?", "ai": "The first man was Adam, created by God from the dust of the ground."},
    {"user": "What did Jesus say about love?", "ai": "Jesus said, 'Love the Lord your God with all your heart and with all your soul and with all your mind. This is the first and greatest commandment. And the second is like it: Love your neighbor as yourself.'"},
    {"user": "Who built the ark?", "ai": "Noah built the ark, as God commanded him, to save his family and the animals from the great flood."},
    {"user": "What is the shortest verse in the Bible?", "ai": "The shortest verse in the Bible is John 11:35: 'Jesus wept.'"},
    {"user": "Who was Moses?", "ai": "Moses was a prophet chosen by God to lead the Israelites out of slavery in Egypt and to deliver the Ten Commandments."},
    {"user": "Who betrayed Jesus?", "ai": "Judas Iscariot, one of the twelve disciples, betrayed Jesus for thirty pieces of silver."},
    {"user": "What is the golden rule?", "ai": "The golden rule is 'So in everything, do to others what you would have them do to you, for this sums up the Law and the Prophets.' (Matthew 7:12)."},
    {"user": "Who defeated Goliath?", "ai": "David, a young shepherd boy, defeated the giant Philistine warrior Goliath with a sling and a stone."}
]

# We duplicate and shuffle the data to make a slightly larger dataset for the model to memorize the format
dataset = []
for _ in range(50):
    for pair in qa_pairs:
        dataset.append(pair)

with open("instruct.json", "w") as f:
    json.dump(dataset, f, indent=4)
    
print(f"Generated {len(dataset)} instruction pairs in instruct.json")
