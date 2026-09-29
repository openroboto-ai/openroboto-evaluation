"""XML-only subset of Robosuite Arena used by its unchanged TableArena class.

The pinned arena has no defaults/includes. Preserve Robosuite's asset path
resolution and collision coloring without importing its simulation stack.
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def array_to_string(values):
    return " ".join(str(value) for value in values)


def string_to_array(value):
    return np.array([float(x) for x in value.split(" ")])


def xml_path_completion(value):
    return str(Path(__file__).parent / "robosuite/assets" / value)


class Arena:
    def __init__(self, filename):
        self.root = ET.parse(filename).getroot()
        if self.root.find("default") is not None or self.root.find("include") is not None:
            raise ValueError("Pinned TableArena unexpectedly needs defaults/include expansion")
        self.worldbody = self.root.find("worldbody")
        self.bottom_pos = np.zeros(3)
        self.floor = self.worldbody.find("./geom[@name='floor']")
        for element in self.root.findall("./asset/*[@file]"):
            element.set("file", str((Path(filename).parent / element.get("file")).resolve()))
        for geom in self.worldbody.iter("geom"):
            if geom.get("group") in (None, "0") and geom.get("name") != "floor":
                geom.set("rgba", "0.5 0.5 0 1")
                geom.attrib.pop("material", None)

    def get_xml(self):
        return ET.tostring(self.root, encoding="unicode")
