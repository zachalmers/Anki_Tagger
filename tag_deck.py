import os, sys
import shutil
import zipfile
import pandas as pd
from anki.collection import Collection

HIGH_RELEVANCE_CUTOFF = 70
MEDIUM_RELEVANCE_CUTOFF = 40
REMOVE_RELEVANCE_CUTOFF = 10

def main(card_path, anki_apkg):

    # Load the csv file into a DataFrame
    df = pd.read_csv(card_path)
    df = df.fillna(0)

    # Group by 'guid' and keep only the row with the highest 'score' for each group
    df = df.loc[df.groupby('guid')['score'].idxmax()]

    # Unzip the .apkg file to a temporary folder
    with zipfile.ZipFile(anki_apkg, 'r') as zip_ref:
        zip_ref.extractall("temp_folder")

    # The main .anki2 database file should have been extracted now
    # Look for the .anki21 file in the temp_folder
    anki2_file = [f for f in os.listdir("temp_folder") if f.endswith(".anki21")][0]

    # Initialize a new collection
    col = Collection(os.path.join("temp_folder", anki2_file))
    tagged = set()

    # Iterate through all cards
    # For each row in the DataFrame

    for index, row in df.iterrows():

        guid = row['guid']
        tag = row['tag']
        score = int(row['score'])

        if score >= HIGH_RELEVANCE_CUTOFF:
            try:
                note_id,note_tags = col.db.all("SELECT id, tags FROM notes WHERE guid = ?",guid)[0]
                new_tag = note_tags + " " + tag+"::1_highly_relevant" + " "
                col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                tagged.add(guid)
            except:
                print(f"guid not found for card: {row['card']}")

        if score < HIGH_RELEVANCE_CUTOFF and score >= MEDIUM_RELEVANCE_CUTOFF:
            try:
                note_id,note_tags = col.db.all("SELECT id, tags FROM notes WHERE guid = ?",guid)[0]
                new_tag = note_tags + " " + tag+"::2_somewhat_relevant" + " "
                col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                tagged.add(guid)
            except:
                print(f"guid not found for card: {row['card']}")

        if score < MEDIUM_RELEVANCE_CUTOFF and score >= REMOVE_RELEVANCE_CUTOFF:
            try:
                note_id,note_tags = col.db.all("SELECT id, tags FROM notes WHERE guid = ?",guid)[0]
                new_tag = note_tags + " " + tag+"::3_minimally_relevant" + " "
                col.db.execute("UPDATE notes set tags = ? where id = ?", new_tag, note_id)
                tagged.add(guid)
            except:
                print(f"guid not found for card: {row['card']}")

    # Save the collection
    col.close()

    # Re-create the .apkg file
    with zipfile.ZipFile(anki_apkg, 'w') as zip_ref:
        for filename in os.listdir("temp_folder"):
            zip_ref.write(os.path.join("temp_folder", filename), arcname=filename)

    # Clean up the temporary folder
    print(f"Tagged {len(tagged)} cards. Process Complete")
    shutil.rmtree("temp_folder")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: tag_deck.py <cards.csv> <anki_deck.apkg>")
        sys.exit(1)
    card_path = sys.argv[1]
    anki_apkg = sys.argv[2]
    main(card_path, anki_apkg)
