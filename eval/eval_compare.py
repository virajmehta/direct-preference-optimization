import csv
import openai
import os
from dotenv import load_dotenv, find_dotenv

input_csv_filename_0 = "wandb_export_2023-08-23T22_37_10.193-04_00.csv"
input_csv_filename_1 = "wandb_export_2023-08-23T22_39_02.909-04_00.csv"
output_csv_filename_0 = "compare_eval_result_reference.csv"
output_csv_filename_1 = "compare_eval_result_policy.csv"

output0 = []
output1 = []

load_dotenv(find_dotenv())
MODEL = "gpt-4"
openai.api_key = os.environ.get("OPENAI_API_KEY")

with open(input_csv_filename_0, 'r') as csvinput0:
    with open(input_csv_filename_1, 'r') as csvinput1:
        with open(output_csv_filename_0, 'w') as csvoutput0:
                with open(output_csv_filename_1, 'w') as csvoutput1:
                    reader0 = csv.reader(csvinput0)
                    reader1 = csv.reader(csvinput1)
                    writer0 = csv.writer(csvoutput0, lineterminator='\n')
                    writer1 = csv.writer(csvoutput1, lineterminator='\n')
                    
                    # extracting field names through first row
                    fields0 = next(reader0)
                    fields0.append('vs_policy_1')
                    fields1 = next(reader1)
                    fields1.append('vs_policy_0')

                    output0.append(fields0)
                    output1.append(fields1)

                    # get index of wanted field
                    prompt_index = fields0.index("prompt")
                    sample_index = fields0.index("sample")

                    for i, (row0, row1) in enumerate(zip(reader0, reader1)):
                    # extracting each data row one by one
                        prompt = row0[prompt_index]
                        prompt_length = row0[prompt_index].__len__()
                        sample0 = row0[sample_index][prompt_length+1:]
                        sample1 = row1[sample_index][prompt_length+1:]

                        # reference paper for message: https://arxiv.org/pdf/2306.05685.pdf
                        response = openai.ChatCompletion.create(
                            model=MODEL,
                            messages=[
                                {"role": "system", "content": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie."},
                                {"role": "user", "content": "[User Question]\n" + prompt + "\n" + "[The Start of Assistant A's Answer]\n" + sample0 + "\n" + "[The End of Assistant A's Answer]\n" + "[The Start of Assistant B's Answer]\n" + sample1 + "\n" + "[The End of Assistant B's Answer]\n"},
                            ],
                            temperature=0,
                        )
                        if response["choices"][0]["message"]["content"] == "A":
                            row0.append('Win')
                            row1.append('Lose')
                        elif response["choices"][0]["message"]["content"] == "B":
                            row0.append('Lose')
                            row1.append('Win')
                        else:
                            row0.append('Tie')
                            row1.append('Tie')
                        
                        output0.append(row0)
                        output1.append(row1)


                    writer0.writerows(output0)
                    writer1.writerows(output1)