import csv
import openai
# import json
# from dotenv import load_dotenv

input_csv_filename = "wandb_export_2023-08-23T13_45_36.658-04_00.csv"
output_csv_filename = "jeopardy_eval_result.csv"

#------- JSON -------
# json_filename = "test_jeopardy_data.json"
# f = open(json_filename)
# test_jeopardy_data = json.load(f)

fields = []
output = []
null_threshold = 0.01

MODEL = "gpt-3.5-turbo"
openai.api_key = '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'

with open(input_csv_filename, 'r') as csvinput:
    with open(output_csv_filename, 'w') as csvoutput:
        # creating a csv reader object
        reader = csv.reader(csvinput)
        writer = csv.writer(csvoutput, lineterminator='\n')
        
        # extracting field names through first row
        fields = next(reader)
        fields.append('evaluation')

        # add the fields to output
        output.append(fields)

        # get index of wanted field
        prompt_index = fields.index("prompt")
        sample_index = fields.index("sample")
        null_index = fields.index("null_prob")
        correct_answer_index = fields.index("correct_answer")
    
        # extracting each data row one by one
        for row in reader:

            #------- JSON -------
            # prompt = row[prompt_index].split(':')[-1].strip()
            # correct_answer = [obj for obj in test_jeopardy_data if obj.get('question')==prompt][0].get('answer')

            if float(row[null_index]) <= null_threshold:
                prompt_length = row[prompt_index].__len__()
                sample = row[sample_index][prompt_length+1:]
                row[sample_index] = sample
                correct_answer = row[correct_answer_index]

                response = openai.ChatCompletion.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are a judge on whether an answer to jeopardy is correct. Answer 'Yes' or 'No' is sufficient."},
                        {"role": "user", "content": "Given the correct answer is: " + correct_answer + ". Can you check if the following answer to the question is correct:" + sample},
                    ],
                    temperature=0,
                )

                #------- JSON -------
                # data = {
                #     "sample": sample,
                #     "correct_answer": correct_answer,
                #     "judge": response,
                # }

                row.append(response)
                output.append(row)
            else:
                response = "Null"
                row.append(response)
                output.append(row)

        
        writer.writerows(output)

#------- JSON -------
# with open("eval_jeopardy.json", "w") as outfile:
#     json.dump(rows, outfile)