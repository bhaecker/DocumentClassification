from copy import deepcopy
import json
import os
import time
import traceback
import shutil


#data_path1 = r"C:\Users\anna_\OneDrive\Documents\ActiveLearning\AL_REST\test_data\has_partial_WS-test"
#data_path2 = r"C:\Users\anna_\OneDrive\Documents\ActiveLearning\AL_REST\test_data\has_complete_WS-test"
#reformated_path = r"C:\Users\anna_\OneDrive\Documents\ActiveLearning\AL_REST\test_data\reformated"

# TODO handle multiple human annotation -> delete additional human annotation

data_path1 = r"D:\ActiveLearning\2020-11-13 19-00\has_partial_human_annotation"
data_path2 = r"D:\ActiveLearning\2020-11-13 19-00\has_complete_human_annotation"
reformated_path = r"D:\ActiveLearning\2020-11-13 19-00\reformated"

#put log into:
#D:\ActiveLearning\2020-11-13 19-00\reformated\reformated_log.txt


segment_labels_still_other = {}
document_labels_still_other = {}
segments_not_on_all_pages = {}
segment_type_None = {}
segment_label_Empty = {}


VERBOSE = True
start_time = time.time()

#### run program
def run():
    copy_data_to_reformated_folder()
    delete_double_annotation()
    change_target_collection_to_partial()
    reformat_document_labels()
    reformat_segment_labels()
    check() 
    print_stats()
################


def copy_data_to_reformated_folder():
    try:
        shutil.rmtree(reformated_path+"\has_partial_human_annotation") 
        shutil.rmtree(reformated_path+"\has_complete_human_annotation") 
    except:
        print("reformated folders do not exist. not deleted.")


    shutil.copytree(data_path1,os.path.join(reformated_path,"has_partial_human_annotation"))
    shutil.copytree(data_path2, os.path.join(reformated_path,"has_complete_human_annotation"))

def change_target_collection_to_partial():
    back_to_partial = [
    "ff2bf2b0497ec43da6bb3685d61c6fb6",
    "7fce5d504c7b40c7adf72d76220223ed",
    "4df67f70accab852ab5be2b7a2b84fe0",
    "15c5a2e0bddc94b0bc50f949ddefeba3",
    "990ec8058916b3c7ad3201cfe2165e5b",
    "ac312c8e22ef531e78aeefaddeed1006",
    "c367a3f6fed373ce56922e3ffa30de6a",
    "20c2658c5dae9c629ccac6227b3fb427",
    "21e14183c246f9e8dab1080366272067",
    "42e0b67fe3a0a30d545dd6686b8fd68e",
    "76bc81c767389a20664292943b1ee7e1",
    "9a4ee9a5c247d95e5099dca28f402796",
    "d5cc3685c1b2a4b1ad8cb8ad82c549f4",
    "ea5062170763e82f2b8aa8148d8d7970",
    "c8ccd7caccd3e7134a40954fb7efa684",
    "76e45bcebe66bae932474d9a9315da87",
    "39f600e40635e61fc0e37b87571ef034",
    "b5e59a125f925eacfa3bd49b78f6591c",
    "37c457395c38c3533f280671cf2e91e1",
    "46581009f67ca7fc3ec5520cab07870c",
    "1a2fcf22bbefd5f1ed392c0f7b7eb4d3",
    "6d2e3041e81f27b0187a485bd9400bc8",
    "67cb1e09c985ea30e521452d16bbeee4",
    "8efdd96dae4b1df615b161a7ffcc3178"
    ]
    
    for data_path in [os.path.join(reformated_path,"has_complete_human_annotation"), os.path.join(reformated_path,"has_partial_human_annotation")]:
        data_dirs = [pos_json for pos_json in os.listdir(data_path) if pos_json.endswith('.json')]
        for filename in data_dirs:
            d = os.path.join(data_path, filename)
            with open(d) as json_file:
                data = json.load(json_file)
                if filename in back_to_partial:
                    data["target_collection"] = "has_partial_human_annotation"

                    # delete all complete annotation markers   
                    data["annotated_by"]["annotators_complete"] = []
                    data["annotated_by"]["segment_boundary"] = []
                    data["annotated_by"]["segment_label"] = []
                    data["annotated_by"]["segment_type"] = []
                    
    
                    # save to file
                    print("write to file",d)
                    with open(os.path.join(reformated_path,"has_partial_human_annotation",filename), 'w') as outfile:
                        json.dump(data, outfile)

    # remove file from has_complete_human_annotation 
    for data_path in [os.path.join(reformated_path,"has_complete_human_annotation")]:
        for filename in back_to_partial:
            filename += ".json"
            d = os.path.join(data_path, filename)
            if os.path.isfile((os.path.join(reformated_path,"has_complete_human_annotation",filename))):
                os.remove(os.path.join(reformated_path,"has_complete_human_annotation",filename))




def delete_double_annotation():
    to_delete =	{
    "8b63c999ae6b9dc652bbe2fc13c7004a.json": "AW",
    "2b9379b050e60df28dccdf2dbeee7113.json": "BJ",
    "80af4f7c1dc046916969f664e081655c.json": "FR",
    "df39cb773f85fedc3cec1f26c5158414.json": "ME",
    "769a9a24c8fabe1669047c0509616a76.json": "ME",
    "28ff5f5a1afcd1b7a81018edf97a450d.json": "MG",
    "b492573905bdc34b23b5578bb515d0aa.json": "SF",
    "8018de61a4715e60e282444e1a6be7fc.json": "SF",
    "aa38609045beba9c5c448c6b6d6785d6.json": "SF",
    "4a6d0b07b7d71ad0020c8c6cbcf73c28.json": "pp"
    }

    segments_to_remove = []
    # delete all segments by this annotator
    for data_path in [os.path.join(reformated_path,"has_complete_human_annotation")]:
        data_dirs = [pos_json for pos_json in os.listdir(data_path) if pos_json.endswith('.json')]
        for filename in data_dirs:

            #print(filename)
            if filename in to_delete:

                d = os.path.join(data_path, filename)
                with open(d) as json_file:
                    data = json.load(json_file)

                    assert(len(data["annotated_by"]["annotators_complete"]) == 2)
                    if(len(data["annotated_by"]["annotators_complete"]) > 2):
                        print("ERROR annotated_by>annotators_complete > 2")

                    # delete from annotators_complete
                    data["annotated_by"]["annotators_complete"].remove(to_delete[filename])

                    for entry in [data["annotated_by"]["segment_boundary"]]:
                        entry_info = entry[0]
                        if entry_info["name"] == to_delete[filename]:
                            data["annotated_by"]["segment_boundary"].remove(entry_info)
                    
                    for entry in [data["annotated_by"]["segment_label"]]:
                        entry_info = entry[0]
                        if entry_info["name"] == to_delete[filename]:
                            data["annotated_by"]["segment_label"].remove(entry_info)
                    
                    for entry in [data["annotated_by"]["segment_type"]]:
                        entry_info = entry[0]
                        if entry_info["name"] == to_delete[filename]:
                            data["annotated_by"]["segment_type"].remove(entry_info)
                               
                    del data["annotator_unconfirmed"][to_delete[filename]]
                    
                            
                    
                    # ----------------- segments -----------------------
                    for page in data["pages"]:
                        for segment in page["segments"]:

                            segment_categories = ["segment_label","segment_type","segment_boundary"]

                            for x in segment_categories:
                                for item in segment["annotation_groups"][x]["annotations"]:

                                    if item["annotator_type"] == "human":

                                        #print("id", item["annotation_id"])
                                        if item["annotation_id"] == "0":
                                            print(item)

                                        if item["annotator_id"] == to_delete[filename]:
                                            #print("append", item["annotation_id"])
                                            segments_to_remove.append(segment)

                                        
                    # remove invalid segments
                    for s in segments_to_remove:
                        for page in data["pages"]:
                            if s in page["segments"]:
                                page["segments"].remove(s)
                                #print(s)
                                break


                    #print("check")
                    # check 
                    for page in data["pages"]:
                        for segment in page["segments"]:

                            segment_categories = [segment["annotation_groups"]["segment_label"]["annotations"], 
                                                segment["annotation_groups"]["segment_type"]["annotations"], 
                                                segment["annotation_groups"]["segment_boundary"]["annotations"]]

                            for item in segment_categories:
                                #print(item)
                                if len(item) > 0:
                                    item = item[0]
                                    if item["annotator_type"] == "human":
                                        #print("id", item["annotation_id"], item["annotator_id"])
                                        assert((item["annotator_id"] == to_delete[filename]) == False)
                                #else:
                                    #print(item)


                    # save to file
                    with open(os.path.join(reformated_path,data["target_collection"],filename), 'w') as outfile:
                        json.dump(data, outfile)       

    # remove from "annotated_by" "annotators_complete"
    # remove from all segments
    # assert: 
    # that other annotator is still there 
    # and deleted annotator is gone
    # no more double
    # save


with open(r"C:\Users\anna_\OneDrive\Documents\ActiveLearning\AL_REST\docker\finetuned_annotation_types.json", "r", encoding="utf-8") as fp:
    finetuned_annotation_types = json.load(fp)

# returns value if no synonym found
def get_synonym_document_label(value):
    replace_labels = [('1_Angebot', 'Angebot'),
                  ('22_1_Behoerdendokument', 'Gesetzesdokument'),
                  ('22_1_Behoerdendokument', 'Anwaltsschreiben'),
                  ('23_1_Technischer_Plan', 'Technischer Plan'),
                  ('23_1_Technischer_Plan', 'Plan'),
                  ('23_1_Technischer_Plan', 'Zeichnung'),
                  ('23_1_Technischer_Plan', 'Bauplan'),
                  ('23_1_Technischer_Plan', 'Technische Zeichnung'),
                  ('23_1_Technischer_Plan', 'technischer Plan'),
                  ('23_1_Technischer_Plan', 'Skizze'),
                  ('23_1_Technischer_Plan', 'technische Zeichnung'),
                  ('23_1_Technischer_Plan', 'Bauskizze'),
                  ('23_1_Technischer_Plan', 'Architektenplan'),
                  ('22_3_Personaldokument', 'Mitarbeiterinformation'),
                  ('22_3_Personaldokument', 'Bewerbung'),
                  ('22_3_Personaldokument', 'Bewerbungsschreiben'),
                  ('8_Werbung', 'Prospekt'),
                  ('23_2_Pruefbericht', 'Prüfbericht'),
                  ('is_other', 'undefined'),
                  ('is_other', 'Undefiniert'),
                  ('is_other', 'undefiniert'),
                  ('is_other', 'Leer'),
                  ('is_other', 'leer'),
                  ('is_other', 'Nicht identifizierbar'),
                  ('is_other', 'Sonstiges'),
                  ('is_other', 'Undefinierr')]

    synonym = [item for item in replace_labels if item[1] == value]
    if len(synonym) > 0:
        synonym = synonym[0][0]
        #print(value,"->",synonym)

    else:
        synonym = value
    return synonym


# returns value if no synonym found
def get_synonym_segment_label(value):
    replace_labels = [
                  ('Abrede-Block', 'Abredeblock'),
                  ('Akteurliste', 'Akteursliste'), # changed valid name: Akteurliste
                  ('Akteurliste', 'Teilnehmer'),
                  ('Akteurliste', 'Teilnehmerliste'),
                  ('Anrede-Block', 'Anredeblock'),
                  #('Angebotsnummer', 'Angebotnummer'),
                  #('Angebotsnummer', 'Angebot Nummer'),
                  #('Angebotsnummer', 'Angebot Numme'),
                  #('Angebotsnummer', 'Angebotsnummer '),
                  ('Auflistung', 'Aufzählungsliste'),
                  ('Auflistung', 'Aufzählungen'),
                  ('Auflistung', 'Liste'),
                  ('Auflistung', 'Aufzählung'),
                  ('Auftragsdaten', 'Auftragsmetadaten'),
                  ('Bankverbindung', 'Bankdaten'),
                  ('Bankverbindung', 'Kontodaten'),
                  ('Bankverbindung', 'Bankverbindun'),
                  ('Bankverbindung', 'Bankverbindung '),
                  ('Bankverbindung', 'Bankinformation'),
                  ('Datum und Ort', 'Ort und Datum'),
                  ('Datum', 'Datun'),
                  ('Datum und Ort', 'Ort, Datum'),
                  ('Fließtext', 'FLießtext'),
                  ('Fließtext', 'Fliestext'),
                  ('Kalkulation', 'Kalkualtion'),
                  ('Kalkulation', 'Berechnnug'),
                  ('Kalkulation', 'Berechnung'),
                  ('Kalkulation', 'Preiskalkulation'),
                  ('Kontakt-Informationen', 'Kontaktdaten'),
                  ('Kontakt-Informationen', 'Kontaktinformation'),
                  ('Kontakt-Informationen', 'Kontakt-Informationen'),
                  #('Kundennummer', 'Kundenummer'), # wird Referenznummer
                  #('Kundennummer', 'Kundennummer'),
                  ('Leistungsbeschreibung', 'Leistungsbescheibung'),
                  ('Leistungsbeschreibung', 'Leistungsberschreibung'),
                  ('Leistungsgegenstand', 'Leistunsgegenstand'),
                  ('Leistungsgegenstand', 'Leistungsgegenstand '),
                  ('Leistungsgegenstand', 'leistungsgegenstand'),
                  ('Lieferbedingungen', 'Lieferbedinungen'),
                  #('Rechnungsnummer', 'Rachnungsnummer'),
                  ('Seiteninformation', 'Seiteninformationen'),
                  ('Überschrift', 'Übeschrift'),
                  ('Überschrift', 'Übeschriften'),
                  ('Unternehmensinformation', 'Unternehmensinformation'),
                  ('Unternehmensinformation', 'Unternemensinformation'),
                  ('Unternehmensinformation', 'Unternehmensinformationen'),
                  ('Unternehmensinformation', 'Unternehmensdaten'),
                  ('Unternehmensinformation', 'Unternehmensinfo'),
                  ('Unternehmensinformation', 'Unternemensinformation'),
                  ('Unternehmensinformation', 'Unternehmensinformaiton'),
                  ('Unternehmensinformation', 'Unternehemensinformation'),
                  #('Unterschriftenzusatzinfo', 'UZ'), #falsche annotationen
                  ('Unterschriftenzusatzinfo', 'Unterschritenzusatzinfo'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftenzusatzinfo'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftszusatzinformation'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftenzusatz'),
                  ('Unterschriftenzusatzinfo', 'Unterschiftenzusatzinfo'),
                  ('Zahlungskonditionen', 'Zahlungsbedingung'),
                  ('Zahlungskonditionen', 'Zahlungsbedingungen'),
                  ('Zahlungskonditionen', 'Zahlungsdaten'),
                  ('Zahlungskonditionen', 'Zahlungsinformationen'),
                  ('Zahlungskonditionen', 'Zahlungskonditionen '),
                  ('Referenznummer', 'Angebot Numme'),
                  ('Referenznummer', 'Angebot Nummer'),
                  ('Referenznummer', 'Angebotnummer'),
                  ('Referenznummer', 'Angebotsnummer'),
                  ('Referenznummer', 'Kundennummer'),
                  ('Referenznummer', 'Auftragsnummer'),
                  ('Referenznummer', 'Bestellnummer'),
                  ('Referenznummer', 'Rechnungsnummer'),
                  ('Referenznummer', 'Rachnungsnummer'),
                  ('Referenznummer', 'Kundenummer'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftenzusatzinformation'),
                  ('Unterschriftenzusatzinfo', 'Unterschiftenzusatzinfo'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftszusatzinformation'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftzusatzinformation'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftzusatzinfo'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftentzusatzinformation'),
                  ('Unterschriftenzusatzinfo', 'Unterschriftenzusatzinfo '), 
                  ('Unterschrift','Unterschrift'),
                  ('Zeichnung', 'Bauplan'),
                  ('Zeichnung', 'Plan'),
                  ('Zeichnung', 'Skizze'),
                  ('Zeichnung', 'Zeichnung'),
                  ('Zeichnung', 'Technischer Plan'),
                  ('is_other', 'Sonstiges')]


    synonym = [item for item in replace_labels if item[1] == value]
    if len(synonym) > 0:
        synonym = synonym[0][0]
    else:
        synonym = value

    #print("synonym", value, "->", synonym)
    return synonym


def is_contained_in_other_segment(same_page_segments, segment_boundary_to_check, id):
    for segment in same_page_segments:
        for segment_boundary in segment["annotation_groups"]["segment_boundary"]["annotations"]:
            if segment_boundary["annotator_type"] == "human" and segment["segment_id"] != id: 
                #print("boundary ", segment_boundary["annotation_vector"])
                #print("seg to check", segment_boundary_to_check)
                b = segment_boundary["annotation_vector"]

                if(
                # check x min
                segment_boundary_to_check[0] >= b[0] and
                # check y min
                segment_boundary_to_check[1] >= b[1] and
                # check x max
                segment_boundary_to_check[2] <=b[2] and
                # check y max
                segment_boundary_to_check[3] <= b[3]
                ):
                    #print(id, segment_boundary_to_check," is inside",b)
                    return True


    #print("do not remove segment: ", id)
    return False

def document_label2index(str):
    labels = finetuned_annotation_types["document_label"]["labels"]
    return labels.index(str)


def segment_label2index(str):
    labels = finetuned_annotation_types["segment_label"]["labels"]
    return labels.index(str)


def index2document_label(idx):
    return finetuned_annotation_types["document_label"]["labels"][idx]


def index2segment_label(idx):
    return finetuned_annotation_types["segment_label"]["labels"][idx]


def is_valid_document_label(str):
    valid_labels = deepcopy(finetuned_annotation_types["document_label"]["labels"])
    valid_labels.remove("7_1_FormblattK7")
    valid_labels.remove("7_2_FormblattK3")
    valid_labels.remove("7_3_FormblattK4")
    valid_labels.remove("19_Subunternehmererklaerung")
    valid_labels.remove("20_1_Pass")
    valid_labels.remove("20_2_Fahrzeugpapiere")
    valid_labels.remove("20_3_Kontokarte")
    valid_labels.remove("21_1_Kontaktliste")
    valid_labels.remove("21_2_Visitenkarte")
    valid_labels.remove("22_2_Lieferschein")
    valid_labels.remove("22_4_Urkunde")
    return str in valid_labels

def is_valid_segment_label(str):
    valid_labels = deepcopy(finetuned_annotation_types["segment_label"]["labels"])
    valid_labels.remove("Anlagenliste")
    valid_labels.remove("Geschwaerzt")
    return str in valid_labels


#def remove_human_annotation_from_segment(segment, annotator_id):
#    segment_copy = copy.deepcopy(segment)
#    for x in ["segment_boundary", "segment_label", "segment_type"]:
#    
#        for item in segment["annotation_groups"][x]["annotations"]:
#                if segment_boundary["annotator_id"] == annotator_id:
#                    segment_copy["annotation_groups"][x]["annotations"].remove(item)
#
#
#    return segment_copy



# also soll ich die base predictions auch löschen? die sowieso nicht predicted werden
# das ist dann sowieso 0 ...

# change human annotations
# valid class -> set 1 in vector
# invalid class -> set 0 in vector and put class in other or leave is_other annotation

#-------------------------------------------------------------------------------------------------------
# --------------------------------------- document_label -----------------------------------------------
#-------------------------------------------------------------------------------------------------------
def reformat_document_labels():
    global document_labels_still_other

    print("-----------------document_label-----------------")

    # 1. change finetuned annotation types
    # extend vectors
    # if is_other is a valid label: set 1 and remove is_other label
    for data_path in [os.path.join(reformated_path,"has_complete_human_annotation"), os.path.join(reformated_path,"has_partial_human_annotation")]:
        data_dirs = [pos_json for pos_json in os.listdir(data_path) if pos_json.endswith('.json')]
        for filename in data_dirs:
            d = os.path.join(data_path, filename)
            with open(d) as json_file:
                data = json.load(json_file)
                # ----------------- document_label -----------------------
                for item in data["document_label"]["annotations"]:
                    if item["annotator_type"] == "human":
                        #print(item["annotation_vector"])
                        # extend vectors to match length of document labels to match document labels in finetuned_annotation_types add 2 more zeros (23_1_Technischer_Plan, 23_2_Pruefbericht)
                        
                        
                        if len(item["annotation_vector"]) < len(finetuned_annotation_types["document_label"]["labels"]):
                            diff = len(finetuned_annotation_types["document_label"]["labels"]) - len(item["annotation_vector"])
                            item["annotation_vector"].extend([0]*diff)

                        assert(len(item["annotation_vector"]) == len(finetuned_annotation_types["document_label"]["labels"]))

                        is_other_value = item["is_other"].strip()

                        # CASE 1: is_other can be assigned to a valid class
                        # set 1 in vector if is_other is in valid class
                        # remove is_other label
                        document_label = get_synonym_document_label(is_other_value)
                        if document_label != None and is_valid_document_label(document_label):
                            if VERBOSE:
                                print("CASE 1.1: is_other (",is_other_value,") can be assigned to a valid class:", document_label ,data["document_id"])
                            item["annotation_vector"][document_label2index(document_label)] = 1
                            item["is_other"] = ""

                        # CASE 2: is_other cannot be assigned to a valid class
                        # leave as-is
                        elif is_other_value != "": 
                            if VERBOSE:
                                print("CASE 1.2: is_other (",is_other_value,") cannot be assigned to a valid class",data["document_id"])
                            if document_label == "is_other":
                                # assign synonym
                                is_other_value = document_label
                                item["is_other"] = document_label

                            document_labels_still_other[is_other_value] = item["annotator_id"] 



                        # CASE 3: class selected in annotation vector but not a valid class anymore
                        # set 0 and put class in is_other if vector indicates invalid class
                        if sum(item["annotation_vector"]) == 1:
                            document_class_idx = item["annotation_vector"].index(1)
                        
                            if not is_valid_document_label(index2document_label(document_class_idx)):
                                if VERBOSE:
                                    print("CASE 1.3: class selected in annotation vector (",index2document_label(document_class_idx), ") but not a valid class anymore",data["document_id"])
                                # set vector to 0
                                item["annotation_vector"][document_class_idx] = 0
                                #put in is_other
                                item["is_other"] = index2document_label(document_class_idx)

                                #print(item["annotation_vector"])
                                #print(item["is_other"])
                                document_labels_still_other[index2document_label(document_class_idx)] = item["annotator_id"]

                            #else:
                                #print("OK",index2document_label(document_class_idx))
                        
                        if sum(item["annotation_vector"]) > 1:
                            print("ERROR sum annotation_vector > 0")

                #print("write to file", os.path.join(reformated_path,data["target_collection"],filename))
                with open(os.path.join(reformated_path,data["target_collection"],filename), 'w') as outfile:
                    json.dump(data, outfile)            

#-------------------------------------------------------------------------------------------------------
# --------------------------------------- segment_label -----------------------------------------------
#-------------------------------------------------------------------------------------------------------


# 1. change finetuned annotation types
# extend vectors
# remove sonstiges/sonstiges
# if is_other is a valid label set 1 and remove is_other label
# remove text etc/sonstiges if inside another segment

def reformat_segment_labels():
    global segment_labels_still_other
    global segments_not_on_all_pages
    global segment_type_None
    global segment_label_Empty

    print("-----------------segment_label-----------------")

    segments_to_remove = []

    # segments only in complete human annotation
    for data_path in [os.path.join(reformated_path,"has_complete_human_annotation")]:    
        data_dirs = [pos_json for pos_json in os.listdir(data_path) if pos_json.endswith('.json')]
        for filename in data_dirs:
            d = os.path.join(data_path, filename)

            with open(d) as json_file:
                data = json.load(json_file)
                page_nr = 0
                for page in data["pages"]:
                    page_nr += 1


                    #  print id if page 2 has no human-annotated segments
                    # find documents like: d5cc3685c1b2a4b1ad8cb8ad82c549f4
                    if page_nr == 2:
                        has_human_annotated_segments = False
                        for segment in page["segments"]:
                            for item in segment["annotation_groups"]["segment_label"]["annotations"]:
                                if item["annotator_type"] == "human":
                                    has_human_annotated_segments = True
                                    break

                        if has_human_annotated_segments == False:    
                            print("ERROR human annotated segments not on all pages: ", data["document_id"])
                            segments_not_on_all_pages[data["document_id"]] = data["annotated_by"]["annotators_complete"]


                    for segment in page["segments"]:
                        is_other_value = ""

                        assert(
                            (len(segment["annotation_groups"]["segment_label"]["annotations"]) == 0) or 
                            (segment["annotation_groups"]["segment_label"]["annotations"][0]["annotator_type"] != "human") or 
                            (len(segment["annotation_groups"]["segment_label"]["annotations"]) == 1 and segment["annotation_groups"]["segment_label"]["annotations"][0]["annotator_type"] == "human") or 
                            (len(segment["annotation_groups"]["segment_type"]["annotations"]) == 1 and segment["annotation_groups"]["segment_type"]["annotations"][0]["annotator_type"] == "human") or 
                            (len(segment["annotation_groups"]["segment_boundary"]["annotations"]) == 1 and segment["annotation_groups"]["segment_boundary"]["annotations"][0]["annotator_type"] == "human")
                        )

                        if (len(segment["annotation_groups"]["segment_label"]["annotations"]) == 0):
                            print("ERROR: no segment label annotation", segment["segment_id"], data["document_id"])
                            print(segment)
                            segment_label_Empty[data["document_id"]] = data["annotated_by"]["annotators_complete"]


                        for item in segment["annotation_groups"]["segment_label"]["annotations"]:
                            if item["annotator_type"] == "human":

                                # extend vectors to match length of segment labels to match segment labels in  finetuned_annotation_types (add 9 more zeros ("Fließtext", "Datum", "Auflistung", "Kalkulation", "Referenznummer", "Unternehmensinformation", "Bankverbindung", "Datum und Ort", "Überschrift" ))
                                if len(item["annotation_vector"]) < len(finetuned_annotation_types["segment_label"]["labels"]):
                                    diff = len(finetuned_annotation_types["segment_label"]["labels"]) - len(item["annotation_vector"])
                                    item["annotation_vector"].extend([0]*diff)
                                
                                assert(len(item["annotation_vector"]) == len(finetuned_annotation_types["segment_label"]["labels"]))

                                is_other_value = item["is_other"].strip()

                                # CASE 1: is_other can be assigned to a valid class
                                # set 1 in vector if is_other is in valid class
                                # remove is_other label
                                seg_label = get_synonym_segment_label(is_other_value)
                                if seg_label != None and is_valid_segment_label(seg_label):
                                    if VERBOSE:
                                        print("CASE 2.1: is_other (",is_other_value,") can be assigned to a valid class:",seg_label,data["document_id"])
                                    # set 1 in suitable class
                                    item["annotation_vector"][segment_label2index(seg_label)] = 1
                                    # set is_other to ""
                                    item["is_other"] = ""


                                # CASE 2: is_other cannot be assigned to a valid class
                                # leave as-is (TODO replace with synonym?)
                                elif is_other_value != "" and is_other_value != "is_other": 
                                    if VERBOSE:
                                        print("CASE 2.2: is_other (",is_other_value,") cannot be assigned to a valid class",data["document_id"])
                                    segment_labels_still_other[is_other_value] = item["annotator_id"]
                                    if seg_label == "is_other":
                                        # assign synonym
                                        item["is_other"] = seg_label


                                # CASE 3: class selected in annotation vector but not a valid class anymore
                                # set 0 and put class in is_other if vector indicates invalid class
                                if sum(item["annotation_vector"]) == 1:
                                    segment_class_idx = item["annotation_vector"].index(1)
                                    seg_label = get_synonym_segment_label(index2segment_label(segment_class_idx))
                                
                                    if not is_valid_segment_label(seg_label):

                                        #print("X : ", index2segment_label(segment_class_idx))
                                        if VERBOSE:
                                            print("CASE 2.3: class selected in annotation vector (",index2segment_label(segment_class_idx), ")  but not a valid class anymore",data["document_id"])
                                        # set vector to 0
                                        item["annotation_vector"][segment_class_idx] = 0
                                        #put in is_other
                                        item["is_other"] = index2segment_label(segment_class_idx)

                                        #print(item["annotation_vector"])
                                        #print(item["is_other"])
                                        segment_labels_still_other[index2segment_label(segment_class_idx)] = item["annotator_id"]


                        for segment_boundary in segment["annotation_groups"]["segment_boundary"]["annotations"]:
                            if segment_boundary["annotator_type"] == "human":

                                # X if segment_label=is_other and segment fully inside other segment -> remove 
                                # CHANGED remove if the is_other annotation is invalid segment label
                                # note: at this point we already put valid is_other types in the annotation vector
                                # => so if a segment_label-it_other is NOT "is_other" but some other string
                                # it can be removed 

                                if (is_contained_in_other_segment(page["segments"], segment_boundary["annotation_vector"], segment["segment_id"])
                                    and 
                                    len(segment["annotation_groups"]["segment_label"]["annotations"]) == 1
                                    and
                                    not segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == "is_other"
                                    and
                                    not segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == ""
                                    and 
                                    not is_valid_segment_label(segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"])
                                    ):

                                    print("remove segment with segment_label =",segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"], "inside other segment", segment["segment_id"], "in document", data["document_id"])

                                    # remove segment completely
                                    assert(segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] != "is_other")

                                    segments_to_remove.append(segment)


                                if len(segment["annotation_groups"]["segment_type"]["annotations"]) != 1:
                                    print("ERROR segment_type, annotations has length", len(segment["annotation_groups"]["segment_type"]["annotations"]))
                                    segment_type_None[data["document_id"]] = data["annotated_by"]["annotators_complete"]



                        # find segments where document_type is invalid
                        for segment_type in segment["annotation_groups"]["segment_type"]["annotations"]:
                            if segment_type["annotator_type"] == "human": 
                                if segment_type["is_other"] == None or (segment_type["is_other"] != "is_other" and segment_type["is_other"] != ""):
                                    segment_type_None[data["document_id"]] = data["annotated_by"]["annotators_complete"]
                                    print("segment_type", segment_type)

                        # if segment_type = is_other and segment_label is_other -> remove
                        if (len(segment["annotation_groups"]["segment_type"]["annotations"]) == 1 and
                            len(segment["annotation_groups"]["segment_label"]["annotations"]) == 1 and

                            segment["annotation_groups"]["segment_type"]["annotations"][0]["is_other"] == "is_other" and
                            segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == "is_other" and

                            segment["annotation_groups"]["segment_label"]["annotations"][0]["annotator_type"] == "human" and
                            segment["annotation_groups"]["segment_type"]["annotations"][0]["annotator_type"] == "human"
                            ):

                            print("remove segment with segment_label = is_other and segment_type = is_other", segment["segment_id"],"in document", data["document_id"])
                            assert(segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == "is_other")
                            assert(segment["annotation_groups"]["segment_type"]["annotations"][0]["is_other"] == "is_other")
                            segments_to_remove.append(segment)
                        else:
                            #print("pass or annotation length incorrect", segment)
                            if len(segment["annotation_groups"]["segment_type"]["annotations"]) > 1:
                                for a in segment["annotation_groups"]["segment_type"]["annotations"]:
                                    if a["annotator_type"] == "human":
                                        print("ERROR: more than one annotation, human", segment["annotation_groups"]["segment_type"]["annotations"])

                            if len(segment["annotation_groups"]["segment_label"]["annotations"]) > 1:
                                for a in segment["annotation_groups"]["segment_label"]["annotations"]:
                                    if a["annotator_type"] == "human":
                                        print("ERROR: more than one annotation, human", segment["annotation_groups"]["segment_label"]["annotations"])





                # remove invalid segments
                for s in segments_to_remove:
                    for page in data["pages"]:
                        if s in page["segments"]:
                            page["segments"].remove(s)
                            break


                                
                with open(os.path.join(reformated_path,data["target_collection"],filename), 'w') as outfile:
                    json.dump(data, outfile)



# ------------------ check reformated -----------------------------------------------------------
# vectors ave correct length, no valid types in other, no invalid classes in annotation_vector
# no invalid segments
def check():
    global segment_label_Empty

    for data_path in [os.path.join(reformated_path,"has_partial_human_annotation"), os.path.join(reformated_path,"has_complete_human_annotation")]:
        data_dirs = [pos_json for pos_json in os.listdir(data_path) if pos_json.endswith('.json')]
        for filename in data_dirs:
            d = os.path.join(data_path, filename)
            try:
                with open(d) as json_file:
                    data = json.load(json_file)
                    # ----------------- document_label -----------------------
                    for item in data["document_label"]["annotations"]:
                        if item["annotator_type"] == "human":

                            # assert correct length 
                            try:
                                assert(len(item["annotation_vector"]) == len(finetuned_annotation_types["document_label"]["labels"]))
                            except:
                                print(d)
                                print("ERROR len", len(item["annotation_vector"]), "should be", len(finetuned_annotation_types["document_label"]["labels"]))
                    
                            # assert max. one 1-entry in vector
                            assert(sum(item["annotation_vector"]) <= 1)

                            is_other_value = item["is_other"].strip()
                            is_other_value = get_synonym_document_label(is_other_value)

                            # if 1 in annotation_vector must be valid class
                            if sum(item["annotation_vector"]) == 1:
                                assert(is_valid_document_label(index2document_label(item["annotation_vector"].index(1))))

                            # if sum(annotation_vector) == 0 -> is_other must be invalid class
                            elif sum(item["annotation_vector"]) == 0:
                                assert(is_valid_document_label(is_other_value) == False)



                            back_to_partial = [
                            "ff2bf2b0497ec43da6bb3685d61c6fb6",
                            "7fce5d504c7b40c7adf72d76220223ed",
                            "4df67f70accab852ab5be2b7a2b84fe0",
                            "15c5a2e0bddc94b0bc50f949ddefeba3",
                            "990ec8058916b3c7ad3201cfe2165e5b",
                            "ac312c8e22ef531e78aeefaddeed1006",
                            "c367a3f6fed373ce56922e3ffa30de6a",
                            "20c2658c5dae9c629ccac6227b3fb427",
                            "21e14183c246f9e8dab1080366272067",
                            "42e0b67fe3a0a30d545dd6686b8fd68e",
                            "76bc81c767389a20664292943b1ee7e1",
                            "9a4ee9a5c247d95e5099dca28f402796",
                            "d5cc3685c1b2a4b1ad8cb8ad82c549f4",
                            "ea5062170763e82f2b8aa8148d8d7970",
                            "c8ccd7caccd3e7134a40954fb7efa684",
                            "76e45bcebe66bae932474d9a9315da87",
                            "39f600e40635e61fc0e37b87571ef034",
                            "b5e59a125f925eacfa3bd49b78f6591c",
                            "37c457395c38c3533f280671cf2e91e1",
                            "46581009f67ca7fc3ec5520cab07870c",
                            "1a2fcf22bbefd5f1ed392c0f7b7eb4d3",
                            "6d2e3041e81f27b0187a485bd9400bc8",
                            "67cb1e09c985ea30e521452d16bbeee4",
                            "8efdd96dae4b1df615b161a7ffcc3178"
                            ]
                            if data["document_id"] in back_to_partial:
                                print("data[target_collection]", data["target_collection"])
                                print(d)
                                assert(data["target_collection"] == "has_partial_human_annotation")

                            try:
                                if data["target_collection"] == "has_complete_human_annotation":
                                    assert(len(data["annotated_by"]["annotators_complete"]) == 1)
                            except:
                                print(d)
                                print("ERROR number of annotators != 1", len(data["annotated_by"]["annotators_complete"]))
                    # ----------------- segment_label -----------------------
                    for page in data["pages"]:
                        for segment in page["segments"]:
                            is_other_value = ""
                            for item in segment["annotation_groups"]["segment_label"]["annotations"]:
                                if item["annotator_type"] == "human":

                                    assert(len(item["annotation_vector"]) == len(finetuned_annotation_types["segment_label"]["labels"]))
                                    
                                    # assert max. one 1-entry in vector
                                    assert(sum(item["annotation_vector"]) <= 1)

                                    is_other_value = item["is_other"].strip()
                                    is_other_value = get_synonym_segment_label(is_other_value)

                                    # if 1 in annotation_vector must be valid class
                                    if sum(item["annotation_vector"]) == 1:
                                        class_idx = item["annotation_vector"].index(1)
                                        assert(is_valid_segment_label(index2segment_label(class_idx)))

                                    # if sum(annotation_vector) == 0 -> is_other must be invalid class
                                    elif sum(item["annotation_vector"]) == 0:
                                        if is_valid_segment_label(is_other_value):
                                            print("should be invalid but is valid", is_other_value)
                                        assert(is_valid_segment_label(is_other_value) == False)

                            for segment_boundary in segment["annotation_groups"]["segment_boundary"]["annotations"]:
                                if segment_boundary["annotator_type"] == "human":

                                    if len(segment["annotation_groups"]["segment_label"]["annotations"]) == 1:

                                        # assert sonstiges no label inside other segment
                                        assert(
                                            (is_contained_in_other_segment(page["segments"], segment_boundary["annotation_vector"], segment["segment_id"]) 
                                            and  
                                            not segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == ""
                                            and 
                                            not segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"] == "is_other"
                                            and 
                                            not is_valid_segment_label(segment["annotation_groups"]["segment_label"]["annotations"][0]["is_other"])) 
                                            == False)
                                    else:
                                        if len(segment["annotation_groups"]["segment_label"]["annotations"]) == 0:
                                            segment_label_Empty[data["document_id"]] = data["annotated_by"]["annotators_complete"]
                                        else:
                                            print("WARNING segment_label len = ",len(segment["annotation_groups"]["segment_label"]["annotations"]), segment)


                            # assert no sonstiges-sonstiges segments
                            for annotation in segment["annotation_groups"]["segment_type"]["annotations"]:
                                if annotation["annotator_type"] == "human":
                                    if (is_other_value == "is_other" and annotation["is_other"] == "is_other"):
                                        print("ASSERT FAIL: is_other/is_other segments", segment)
                                        print(data["document_id"])
                                        assert((is_other_value == "is_other" and annotation["is_other"] == "is_other") == False)

                    #print("passed")
            except AssertionError as e:
                    print(type(e))    
                    traceback.print_exc()
                    #print(data["document_id"])
                    #print(segment)



def print_stats():
    print("\n'is_other' document labels:")
    for item in sorted((document_labels_still_other)):
        print(item)

    print("\n'is_other' segment labels: ")
    for item in sorted(segment_labels_still_other):
        print(item)

    print("\nsegments not on all pages:")
    for item in sorted(segments_not_on_all_pages):
        print(item, segments_not_on_all_pages[item])

    print("\nsegment_type_None:")
    for item in sorted(segment_type_None):
        print(item, segment_type_None[item])

    print("\nsegment_label_Empty:")
    for item in sorted(segment_label_Empty):
        print(item, segment_label_Empty[item])


#print("time elapsed (min):", (time.time()-start_time)/60)






############# run program #############
run()







