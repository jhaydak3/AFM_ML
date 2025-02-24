Usage

Usage: ARDFtoHDF5 <file.ARDF> <file.h5> in windows command prompt while in the directory containing the .exe file.

Specification
Create a conversion tool that reads an ARDF file and writes an HDF5 file. This will only support FFM files.
Images will be stored as one 2D dataset for each channel
Force data will be stored as one 3D dataset for each channel.
A concatenation of all note information will be put in the "Note" attribute on the root node.
-----
Thanks David, you put this out at exactly the right time for me. Could you explain how to interpret the /ForceMaps/Segments dataset?

EDIT: In an email reply, David explained:

The ForceMap/Segments attribute (not the dataset) has: Ext, Ret, Away
Ext is extension
Ret is retraction
Away I would assume is a dwell time after retraction but I can check if that doesn’t sound right.

So the first number in the ForceMap/Segments dataset is what data index corresponds to the end of the extension phase, the next is for the end of retraction, etc.

Force maps can have up to 4 segments but yours is only using 3. This conversion tool is [...] giving a 4th value of 0 for the unused segment.