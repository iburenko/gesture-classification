from os import path
import copy
import xml.etree.ElementTree as ET

def template2case_xml(src_xml_path, timeseries, video_filename):
    tree, root = open_xml_file(src_xml_path)
    root = change_source_url(root, video_filename)
    root = adjust_min_max(root, timeseries)
    return tree

def open_xml_file(src_xml_path):
    tree = ET.parse(src_xml_path)
    root = tree.getroot()
    return tree, root

def change_source_url(root, video_filename):
    root[0].attrib["source-url"] = "file:///foo/" + video_filename + "_timeseries.csv"
    return root

def adjust_min_max(root, timeseries):
    for elem in root[0]:
        if elem.tag == "track":
            series_name = elem.attrib["name"]
            if "eyebrow" in series_name:
                ts_min = 0.75
                ts_max = 1.25
            else:
                ts = timeseries[series_name]
                ts = ts[ts != 0]
                ts_min, ts_max = str(ts.min() - 0.5), str(ts.max() + 0.5)
            elem[4].attrib["min"] = str(ts_min)
            elem[4].attrib["max"] = str(ts_max)
    return root

def handle_eaf_file(eaf_file, template_eaf, eaf_filename, video_filename):
    modified_template = modify_templtae_eaf(template_eaf, eaf_filename, video_filename)
    modified_header = modified_template[0]
    tree, root = open_xml_file(eaf_file)
    header = root[0]
    for elem in modified_header:
        if elem.tag == "MEDIA_DESCRIPTOR":
            header[0].attrib = elem.attrib
            openpose_elem = copy.deepcopy(elem)
            openpose_elem.attrib["MEDIA_URL"] = \
                openpose_elem.attrib["MEDIA_URL"][:-4] + "_openpose_black.mp4"
            openpose_elem.attrib["RELATIVE_MEDIA_URL"] = path2openposepath(elem.attrib["RELATIVE_MEDIA_URL"])
            header.append(openpose_elem)
        elif elem.tag == "LINKED_FILE_DESCRIPTOR":
            header.append(elem)
    return tree

def path2openposepath(video_path):
    video_basename = path.basename(video_path)
    return "../video with OpenPose output/" + video_basename[:-4] + "_openpose_black.mp4"
    
def modify_templtae_eaf(template_eaf, eaf_filename, video_filename):    
    video_filename = path.basename(video_filename)[:-4]
    _, root = open_xml_file(template_eaf)
    header = root[0]
    for elem in header:
        if elem.tag == "MEDIA_DESCRIPTOR":
            src_rel_media_url = elem.attrib["RELATIVE_MEDIA_URL"]
            dst_media_url = "file:///foo/" + video_filename + ".mp4"
            dst_rel_media_url = \
                change_basename(src_rel_media_url, video_filename + ".mp4")
            elem.attrib["MEDIA_URL"] = dst_media_url
            elem.attrib["RELATIVE_MEDIA_URL"] = dst_rel_media_url
        elif elem.tag == "LINKED_FILE_DESCRIPTOR":
            if elem.attrib["MIME_TYPE"] == "text/xml":
                elem.attrib["LINK_URL"] = "file:///foo/" + eaf_filename + "_tsconf.xml"
                elem.attrib["RELATIVE_LINK_URL"] = "./" + eaf_filename + "_tsconf.xml"
            elif elem.attrib["MIME_TYPE"] == "unknown":
                csv_rel_link = elem.attrib["RELATIVE_LINK_URL"]
                elem.attrib["ASSOCIATED_WITH"] = "file:///foo/" + eaf_filename + ".mp4"
                elem.attrib["LINK_URL"] = "file:///foo/" + eaf_filename + "_timeseries.csv"
                elem.attrib["RELATIVE_LINK_URL"] = \
                    change_basename(csv_rel_link, eaf_filename + "_timeseries.csv")
    return root

def find_xml_field(xml_elem, tag):
    for elem in xml_elem:
        if elem.tag == tag:
            return elem
    return None
        
def change_basename(src_path, target_name):
    return "/".join(src_path.split("/")[:-1] + [target_name])