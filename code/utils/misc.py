import pathlib
import os
def getRelativeFp(fileDunder, pathToAppend):
    filePathParent = pathlib.Path(fileDunder).parent.absolute()
    newFilePath = os.path.join(filePathParent, pathToAppend)
    fpParent = pathlib.Path(newFilePath).parent.absolute()
    if not os.path.exists(fpParent):
        os.makedirs(fpParent)
        print(f"Created directory {fpParent}")
    return newFilePath