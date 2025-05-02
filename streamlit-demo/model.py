import os 
import sys
import pickle

import torch
from torch import nn
from torchtext.data.utils import get_tokenizer



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mulstage_model import CNN_BiLSTM, SecondaryModel

model_dir = "../models"
vocab_dir = "../vocabs"
vocab_file = "vokab_news_traindata_99000.0.pkl"
primary_model_file = "primary_model_news_traindata_99000.0.pth"
secondary_model_file = "secondary_model_news_traindata_99000.0.pth"
label_file = "../encoded/secondary_label_encoding_news_traindata_99000.0.pkl"

special_tokens = ['<unk>', '<pad>']
tokenizer = get_tokenizer('basic_english')

with open(os.path.join(vocab_dir, vocab_file), 'rb') as f:
    vocab = pickle.load(f)

with open(os.path.join(label_file), 'rb') as f:
    label_encoder = pickle.load(f)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

primary_model = CNN_BiLSTM(vocab=vocab, vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=1, pad_idx=vocab['<pad>'])
primary_model.to(device)
primary_model.load_state_dict(torch.load(os.path.join(model_dir, primary_model_file), map_location=device))

secondary_model = CNN_BiLSTM(vocab=vocab, vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=len(label_encoder.classes_), pad_idx=vocab['<pad>'])
secondary_model.to(device)
secondary_model.load_state_dict(torch.load(os.path.join(model_dir, secondary_model_file), map_location=device))



def encode_text(text: str, vocab: dict):
    encoded = [vocab[token] for token in tokenizer(text)]
    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)

def determine_news_validity(title: str, content: str):
    text = title + " " + content
    encoded_text = encode_text(text, vocab)
    primary_model.eval()
    with torch.no_grad():
        output = primary_model(encoded_text)
        prediction = torch.sigmoid(output).item()

    if prediction > 0.5:
        print("Fake News")
        return 1
    else:
        print("Real News")
        return 0
    
def determine_news_category(title: str, content: str):
    text = title + " " + content
    encoded_text = encode_text(text, vocab)
    secondary_model.eval()
    with torch.no_grad():
        output = secondary_model(encoded_text)
        prediction = torch.argmax(output, dim=1).item()

    category = label_encoder.inverse_transform([prediction])[0]
    return category
    

# # fake news example

title = "The 9/11 Commission Didn’t Believe the Government… So Why Should We?"
content = '9/11 Commissioners Admit They Never Got the Full Story\n\nThe 9/11 Commissioners' \
'publicly expressed anger at cover ups and obstructions of justice by the government into a real 9/11' \
'investigation:\n\nThe 9/11 Commission chair said the Commission was “set up to fail”\n\nThe Commission’s' \
'co-chairs said that the CIA (and likely the White House) “obstructed our investigation”\n\n9/11 Commissioner' \
'Timothy Roemer said “We were extremely frustrated with the false statements we were getting”\n\nThe Senior' \
'Counsel to the 9/11 Commission (John Farmer) – who led the 9/11 staff’s inquiry – said “At some level of the' \
'government, at some point in time…there was an agreement not to tell the truth about what happened“. He also' \
'said “I was shocked at how different the truth was from the way it was described …. The tapes told a radically' \
'different story from what had been told to us and the public for two years…. This is not spin. This is not true.' \
'”\n\nNo wonder the Co-Chair of the congressional investigation into 9/11 – Bob Graham – and 9/11 Commissioner' \
'and former Senator Bob Kerrey are calling for either a “PERMANENT 9/11 commission” or a new 9/11 investigation' \
'to get to the bottom of it.\n\nSome examples of obstruction of justice into the 9/11 investigation include:\n\nAn' \
'FBI informant hosted and rented a room to two hijackers in 2000. Specifically, investigators for the Congressional' \
'Joint Inquiry discovered that an FBI informant had hosted and even rented a room to two hijackers in 2000 and that,' \
'when the Inquiry sought to interview the informant, the FBI refused outright, and then hid him in an unknown location,' \
'and that a high-level FBI official stated these blocking maneuvers were undertaken under orders from the White House. As' \
'the New York Times notes:\n\nSenator Bob Graham, the Florida Democrat who is a former chairman of the Senate Intelligence' \
'Committee, accused the White House on Tuesday of covering up evidence ….The accusation stems from the Federal Bureau of' \
'Investigation’s refusal to allow investigators for a Congressional inquiry and the independent Sept. 11 commission to' \
'interview an informant, Abdussattar Shaikh, who had been the landlord in San Diego of two Sept. 11 hijackers.\n\nThe chairs' \
'of both the 9/11 Commission and the Official Congressional Inquiry into 9/11 said that Soviet-style government “minders”' \
'obstructed the investigation into 9/11 by intimidating witnesses (and see this)\n\nThe 9/11 Commissioners concluded that officials from the Pentagon lied to the Commission, and considered recommending criminal charges for such false statements\n\nThe tape of interviews of air traffic controllers on-duty on 9/11 was intentionally destroyed by crushing the cassette by hand, cutting the tape into little pieces, and then dropping the pieces in different trash cans around the building as shown by this NY Times article (summary version is free; full version is pay-per-view) and by this article from the Chicago Sun-Times\n\nAs reported by ACLU, FireDogLake, RawStory and many others, declassified documents shows that Senior Bush administration officials sternly cautioned the 9/11 Commission against probing too deeply into the terrorist attacks of September 11, 2001\n\nBoth the 9/11 Commission Investigation and 9/11 Trials Were Based on Unreliable Evidence Produced by Torture\n\nThe CIA videotaped the interrogation of 9/11 suspects, falsely told the 9/11 Commission that there were no videotapes or other records of the interrogations, and then illegally destroyed all of the tapes and transcripts of the interrogations.\n\n9/11 Commission co-chairs Thomas Keane and Lee Hamilton wrote:\n\nThose who knew about those videotapes — and did not tell us about them — obstructed our investigation.\n\nThe chief lawyer for Guantanamo litigation – Vijay Padmanabhan – said that torture of 9/11 suspects was widespread.\n\nAnd Susan J. Crawford – the senior Pentagon official overseeing the military commissions at Guantánamo told Bob Woodward:\n\nWe tortured Qahtani. His treatment met the legal definition of torture.\n\nIndeed, some of the main sources of information were tortured right up to the point of death.\n\nMoreover, the type of torture used by the U.S. on the Guantanamo suspects is of a special type. Senator Levin revealed that the U.S. used Communist torture techniques specifically aimed at creating false confessions. (and see this, this, this and this).\n\nAnd according to NBC News:\n\nMuch of the 9/11 Commission Report was based upon the testimony of people who were tortured\n\nAt least four of the people whose interrogation figured in the 9/11 Commission Report have claimed that they told interrogators information as a way to stop being “tortured”\n\nOne of the Commission’s main sources of information was tortured until he agreed to sign a confession that he was NOT EVEN ALLOWED TO READ\n\nThe 9/11 Commission itself doubted the accuracy of the torture confessions, and yet kept their doubts to themselves\n\nIf the 9/11 Commissioners themselves doubt the information from the government, why should we believe it?\n\nDelivered by The Daily Sheeple\n\nWe encourage you to share and republish our reports, analyses, breaking news and videos (Click for details).\n\nContributed by of Washington’s Blog.'

# # real news example
# title = 'Truth Stays Hidden if No One Looks for It'
# content = "The problems and ambiguity about Lo Duca’s gambling habits cannot be answered because of a condition within sports that can simply be labeled learned ignorance.\n\nWith reports in The Daily News about illegal bookies stalking Lo Duca at ballparks, Mets General Manager Omar Minaya told The New York Times he had asked his All-Star catcher about whether he gambled on horses — which, as a horse owner, Lo Duca admitted that he did — but did not inquire about any gambling junkets beyond the track.\n\nThis strategic incuriosity is not about Minaya or Lo Duca, but about the general fear alive in sports when anyone is pushed to confront issues that threaten to damage the team, the player or the league.\n\n• Why risk a home run or touchdown catch, why taint an N.B.A. icon or a hockey star-maker with too much knowledge?\n\nThe Yankees did not dare ponder the Balco whispers surrounding the steroid use of Jason Giambi until he plunged into an endless funk last spring. The Yankees tried everything but the Jaws of Life to extricate themselves from Giambi’s deal until, of course, his bat started reconnecting with his old bulky self.\n\nPhoto\n\nBaseball refused to connect the dots between the androstenedione found in Mark McGwire’s locker during the lucrative euphoria of the 1998 great home run chase and its inflatable players until Balco and politics forced Commissioner Bud Selig out of his comfort zone as a cheater protectionist.\n\nBaseball isn’t to be singled out. The N.B.A. once dismissed the unsavory associations Michael Jordan had with the gambling underworld as if they were traveling violations (also never called on his Airness). And in the N.F.L., the Oakland Raiders understood that center Barret Robbins had mental health problems but failed to address them until he went on a tequila binge the day before the Super Bowl in 2003.\n\nAdvertisement Continue reading the main story\n\nWillful denial abounds every time a Terrell Owens resurfaces on a team that swears it is satisfied with whatever homework/background check has been applied to a player’s past patterns of disobedience. Dangerous oblivion emerged in the N.H.L. when the union allowed the agent David Frost — with a history of mentally manipulating young players — to continue as the mentor of Mike Danton until Danton was arrested in a murder-for-hire plot.\n\nNewsletter Sign Up Continue reading the main story Please verify you're not a robot by clicking the box. Invalid email address. Please re-enter. You must select a newsletter to subscribe to. Sign Up You agree to receive occasional updates and special offers for The New York Times's products and services. Thank you for subscribing. An error has occurred. Please try again later. View all New York Times newsletters.\n\nThe pressure to know nothing is intense. Only Kevin Towers, as the Padres’ general manager, has been compelled to purge his conscience by speaking up last year about his knowledge of Ken Caminiti’s steroid use.\n\n“I feel somewhat guilty, because I felt like I knew,’’ Towers told espn.com. “I still don’t know for sure, but Cammy came out and said that he used steroids, and I suspected. Selfishly, the guy was putting up numbers, and I didn’t do anything about it. That’s just the truth.’’\n\nThis is the reality that everyone — including the Mets — has to deal with when deciding how deep to investigate a player’s personal problem.\n\nAll the information clubs use in determining a player’s value to the team — How does he hit on sunny days, cloudy days and rain-delayed days? How does he pitch on six hours sleep and catnaps? — yet they resist looking in the darker corners, beyond the stats.\n\nThis is not about a player’s love trysts, night-crawling habits or marital discord. This is not about voyeur tabloids or saucy dish, but about wanting to know as much about a threat to the sport or the player or the team as possible.\n\n•Major League Baseball has cleared Lo Duca; the Mets have done their self-check as well. Everyone is at ease with the situation, and so is Lo Duca.\n\n“The Mets said it for me,’’ Lo Duca said last week. “I don’t need to say nothing else.’’\n\nAll is well, right? So why fret over Lo Duca? Why probe for details?\n\nBecause teams and leagues have no credibility as sleuths. Because teams and leagues are culprits of learned ignorance. The incurious make lousy detectives."

print(determine_news_validity(title, content))
print(determine_news_category(title, content))