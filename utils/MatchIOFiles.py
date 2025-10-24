import os
'''
Need to create list of matching input and target files by reading from directories
Used as input to the Torch.Dataloader iterator 
'''


def matching_io(sc_path, capi_path):

    try:
        capi_files = os.listdir(capi_path)
        sc_files = os.listdir(sc_path)

    except FileNotFoundError:
        print("Check if CAPI and SC data directories exist")

    capi_files = sorted(capi_files)
    sc_files = sorted(sc_files)

    def get_file_id(file_name):
        lis = file_name.split("-")
        if len(lis) != 4:
            raise ValueError
        return [lis[1], lis[2], lis[3].split(".")[0]]    # [month, day, trial]

    # Gets all the file ID's from the smart cushion directory
    sc_file_id = []
    for f in sc_files:
        try:
            sc_file_id.append(get_file_id(f))   # What is the order its being appended in?

        except ValueError:
            print(f"Invalid SC file name: {f}")
            continue

    matching_datapaths = []

    for f in capi_files:
        try:
            get_file_id(f)

            if get_file_id(f) in sc_file_id:
                capi_datapath = os.path.join(capi_path, f)
                sc_datapath = os.path.join(sc_path, sc_files[sc_file_id.index(get_file_id(f))])

                matching_datapaths.append([capi_datapath, sc_datapath])
        except ValueError:
            print(f"Invalid CAPI file name: {f}")
            continue
    
    return matching_datapaths


# Now we need to find the matching CAPI IDs, and make a new list of the matching pairs

