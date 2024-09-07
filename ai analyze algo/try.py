
import turntoframe
import chat2
import checksiffilepresent



count=1
file=str(count)+'.png'
while checksiffilepresent.check_file_exists("outputs",file):
    chat2.image_needed(count)
    count=count+1
    file=str(count)+'.png'