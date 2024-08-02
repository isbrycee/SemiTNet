import sys,os
import cv2
import linecache
import random

def main(list_info,html_page,root_dir,image_file):
    list_info_fp = open(list_info,"r")
    html_page_fp = open(html_page,"w")
    html_page_fp.write('<html><body>\n')
    html_page_fp.write('<p>\n')
    html_page_fp.write('<table border="1">\n')
    index = 0
    index_lou =0
    #str = linecache.getlines(list_info)
    str = list_info_fp.readlines()
    image_dir = os.path.join(root_dir, image_file)
    #print len(str)
    #random.shuffle(str)
    #print str
    for i,line in enumerate(str):
    #for line in str:
        image = os.path.join(image_dir,line.strip())
        image_display = os.path.join(image_file, line.strip())
        print(image_display)
        image_display_ufo = os.path.join('gem', line.strip())
        image_display_ufo1 = os.path.join('SwinUMamba', line.strip())
        image_display_ufo2 = os.path.join('T-Mamba2D', line.strip())
        image_display_ufo3 = os.path.join('gt', line.strip())
        # print(image_display_ufo)
        #image = os.path.join(image_dir, line)
        # print(image)
        #print type(image)
        if not os.path.exists(image):
            print('continue')
            continue
        index += 1
        #if index>500:
            #break
        img = cv2.imread(image)
        #print img
        #width,height = 1280, 720
        width,height = img.shape[1],img.shape[0]
        html_page_fp.write('<tr>\n')
        html_page_fp.write('<td>image_name:%s,width:%s,height:%s</td>\n' % (line.strip(),width,height))
        #html_page_fp.write('<td><img src="%s" width="1920" height="1080" /></td>\n' % (image))
        #html_page_fp.write('<td><img src="%s" width="1120" height="960" /></td>\n' % (image_display))
        html_page_fp.write('<td><img src="%s" width="800" height="600" /></td>\n' % (image_display))
        html_page_fp.write('<td><img src="%s" width="800" height="600" /></td>\n' % (image_display_ufo))
        html_page_fp.write('<td><img src="%s" width="800" height="600" /></td>\n' % (image_display_ufo1))
        html_page_fp.write('<td><img src="%s" width="800" height="600" /></td>\n' % (image_display_ufo2))
        html_page_fp.write('<td><img src="%s" width="800" height="600" /></td>\n' % (image_display_ufo3))
        html_page_fp.write('</tr>\n')
    html_page_fp.write("</tr>\n</table>\n</p>\n</body>\n</html>")
    print('sum:',index)
    print('miss:',index_lou)
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])




