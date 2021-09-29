import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import datetime
from numpy import linalg as LA
import math
import datetime
import matplotlib.colors
# import seaborn as sns
import copy
# import requests
import os

PATH = "/work"  # fitsデータを保存しているファイルパス
PWD = Path(PATH)
# ACCESS_TOKEN = "XXXXX" #LINEに通知を飛ばすためのアクセストークン
print(os.listdir("./"))

class Basedef:
	def npy_select(message = "ファイル選択"):
		print(message)
		npy_list = list(PWD.glob('*.npy'))
		for i in range(len(npy_list)):
			print(str(i) + ": " + str(npy_list[i]).replace(PATH, ""))
		select_number = input("入力: ")
		if not select_number:
			npy = np.load(str(npy_list[3]), allow_pickle = True)
		else:
			npy = np.load(str(npy_list[int(select_number)]), allow_pickle = True)
		return npy

	def list_transpose(list1):
		new_list = np.array(list1)
		new_list = new_list.reshape((new_list.shape[0], 1))
		new_list.transpose()
		return new_list

	def plot(data, rad = 8, deg = 0):
		x = data[:, 0] #L
		y = data[:, 1] #V
		#logk = data[:,2] #I
		logk = np.log10(data[:, 2])
		max_mass = np.nanmax(logk) #28058.733041(log: 4.44806805708)
		min_mass = np.nanmin(logk) #498.991026 (log: 2.69809273522)
		print(max_mass)
		print(min_mass)
		#plt.figure(figsize=(16, 4), dpi=300)
		plt.figure(figsize = (7, 5), dpi = 300)
		#plt.xlim(90, -90)
		plt.xlim(50, 10)
		#plt.ylim(np.nanmin(y), np.nanmax(y))
		plt.ylim(- 100, 200)
		plt.scatter(x, y, s=0.1, alpha=1, c=logk, vmin=min_mass, vmax=max_mass, cmap='jet')
		#plt.scatter(x, y, s=0.1, alpha=1,c=logk,vmin=0, vmax=5, cmap='jet')
		plt.title("Simulation LV (r,θ)=(%s kpc, %s deg.)" %(rad, deg))
		#plt.title("CfA Observation")
		plt.xlabel("L [deg.]")
		plt.ylabel("V [km/s]")
		plt.colorbar().set_label('I[K]', labelpad = -27, y = 1.05, rotation = 0) #labelpadを大きくすると右に。yを大きくすると上に。
		#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
		dt = datetime.datetime.now()
		image = plt.savefig("(r,θ)=(%s, %s)_%s.png" % (rad, deg, dt), bbox_inches='tight', pad_inches=0)
		plt.clf()
		return image

	def plot2(data): # XYプロット用
		#x = data[:,7]
		#y = data[:,12]
		x = data[:, 2]
		y = data[:, 3]
		#x = data[:,11] #RHO[H/cc]
		#y = data[:,12] #T[K]
		#k = data[:,7] #M_sun
		#logx = np.log10(x)
		#logy = np.log10(y)
		logx = x
		logy = y
		#logk = np.log10(k) #np.log(k)
		#max_mass = np.nanmax(logk) #28058.733041(log: 4.44806805708)
		#min_mass = np.nanmin(logk) #498.991026 (log: 2.69809273522)
		#plt.xlim(-30, 30)
		#plt.ylim(- 30, 30)
		plt.figure(figsize=(6, 5), dpi=300)
		plt.xlim(np.nanmin(logx) - 1, np.nanmax(logx) + 1)
		plt.ylim(np.nanmin(logy) - 1, np.nanmax(logy) + 1)
		plt.scatter(logx, logy, s=0.1, alpha=1)
		#plt.scatter(logx, logy, s=0.1, alpha=1,c=logk,vmin=min_mass, vmax=max_mass, cmap='jet')
		plt.title("Simulation M-T LOG")
		plt.xlabel("M [M_sun]")
		plt.ylabel("T [K]")
		#plt.colorbar().set_label('M_sun', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', pad_inches=0) #, bbox_inches='tight'
		plt.clf()
		return image

	def obspos_plot(data, rad, deg, title=""): #scatterの順番でカラーバーが上手く指定範囲で表示されないことがある
		x = data[:, 0] #X
		y = data[:, 1] #Y
		k = np.log10(data[:, 11]) #RHO
		#k = data[:,11] #RHO
		obs_x = rad * math.cos(math.radians(deg))
		obs_y = rad * math.sin(math.radians(deg))

		plt.figure(figsize=(6, 5), dpi=300)
		plt.xlim(np.nanmin(x) - 1, np.nanmax(x) + 1)
		plt.ylim(np.nanmin(y) - 1, np.nanmax(y) + 1)
		#plt.xlim(- 30, 30)
		#plt.ylim(- 30, 30)
		plt.scatter(x, y, s=0.1, alpha=1,c=k,vmin=np.nanmin(k), vmax=np.nanmax(k), cmap='jet')
		plt.colorbar().set_label('RHO [H/cc]', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		plt.scatter(obs_x, obs_y, s=30,c='red')
		plt.title(title)
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")
		dt = datetime.datetime.now()
		image = plt.savefig("(r,θ)=(%s, %s)_%s.png"%(rad, deg, dt), bbox_inches='tight', pad_inches = 0)
		plt.clf()

	def npy_cut(data, deg_min,deg_max):
		index = np.where((deg_min <= data[:,2]) & (data[:,2] <= deg_max)) #VTLDM用
		cutted_data = data[index]
		dt_now = datetime.datetime.now()
		file_name = "new_VTLDM_("+ str(dt_now) + ").npy"
		np.save(file_name, cutted_data)
		return file_name

	def u2k_cut(name):
		data = Basedef.npy_search(name)
		l_l = data[:,0]
		v_l = data[:,1]
		k_l = data[:,2] - 2.7
		k0_index = np.where(k_l != 0)
		k_l = k_l[k0_index]
		l_l = l_l[k0_index]
		v_l = v_l[k0_index]
		l_l = Basedef.list_transpose(l_l)
		v_l = Basedef.list_transpose(v_l)
		k_l = Basedef.list_transpose(k_l)
		new_LVI = np.concatenate((l_l, v_l, k_l), axis=1)
		dt_now = datetime.datetime.now()
		u2_name = "U2C_NewLVI("+ str(dt_now) + ").npy"
		np.save(u2_name, new_LVI)
		return u2_name

	def npy_search(name):
		npy_list = list(Path(os.getcwd()).glob('*.npy')) #今いるディレクトリ上でnpyデータを検索する
		#npy_list = list(PWD.glob('*.npy')) #フォルダl11-50で検索
		str_list = [str(p) for p in npy_list]
		search_list = [s for s in str_list if name in s]
		npy = np.load(str(npy_list[str_list.index(search_list[0])]), allow_pickle = True)
		return npy

	def dot_search(data, l_min, l_max, v_min, v_max, min_k): #dataは[L,V,K]
		l_index = np.where((data[:, 0] >= l_min) & (data[:, 0] <= l_max))
		data = data[l_index]
		v_index = np.where((data[:, 1] >= v_min) & (data[:, 1] <= v_max))
		data = data[v_index]
		k_index = np.where(data[:, 2] >= min_k)
		data = data[k_index]
		return data

	def double_plot(data, pick_data):
		x = data[:, 0] #X
		y = data[:, 1] #Y
		k = np.log10(data[:, 2]) #RHO
		#k = data[:,11] #RHO
		pick_x = pick_data[:, 0]
		pick_y = pick_data[:, 1]

		plt.figure(figsize=(16, 4), dpi=300)
		plt.xlim(np.nanmax(x), np.nanmin(x))
		plt.ylim(np.nanmin(y), np.nanmax(y))
		plt.scatter(x, y, s=0.1, alpha=1, c=k, vmin=np.nanmin(k), vmax=np.nanmax(k), cmap='jet')
		plt.colorbar().set_label('I [K]', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		plt.scatter(pick_x, pick_y, s=0.1,c='red')
		plt.title("Simulation R8θ65")
		plt.xlabel("L [deg.]")
		plt.ylabel("V [km/s]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
		plt.clf()

	def match_dotsearch(data, xy_data): #pick_data[L,V]のLVに合致するxy_data[L,V,X,Y]のLVXYを返す
		def square_search(lvxy, ma_l, ma_v, l_lim, v_lim): #matchした点の周囲にある点を見つける
			lindex_u = np.where(lvxy[:, 0] <= ma_l + l_lim) #lの上限下限に入るデータ
			lvxyl_u = lvxy[lindex_u]
			lindex_l = np.where(lvxyl_u[:, 0] >= ma_l - l_lim)
			lvxy_lu = lvxyl_u[lindex_l]
			vindex_u = np.where(lvxy_lu[:, 1] <= ma_v + v_lim) #vの上限下限に入るデータ
			lvxyv_u = lvxy_lu[vindex_u]
			vindex_l = np.where(lvxyv_u[:, 1] >= ma_v - v_lim)
			pick_data = lvxyv_u[vindex_l]
			return pick_data

		match_l = []
		for i in tqdm(range(data.shape[0])):
			data_l = data[i, 0] #pickしてきたデータのL
			data_v = data[i, 1] #pickしてきたデータのV

			search_data = square_search(xy_data, data_l, data_v, 1, 10)

			for j in range(search_data.shape[0]):
				mini_match = []
				mini_match.append(search_data[j, 0])
				mini_match.append(search_data[j, 1])
				mini_match.append(search_data[j, 2])
				mini_match.append(search_data[j, 3])
				match_l.append(mini_match)

		match_lvxy = np.array(match_l)
		return match_lvxy

	def double_plotXY(data, pick_data, x_range = 8.5, y_range = 8.5): #dataは[L,V,X,Y]
		x = data[:,2] #X
		y = data[:,3] #Y
		pick_x = pick_data[:,2]
		pick_y = pick_data[:,3]
		plt.figure(figsize=(6, 5), dpi=300)
		plt.xlim(- x_range - 1, x_range + 1)
		plt.ylim(- y_range - 1, y_range + 1)
		plt.scatter(x, y, s=0.1, alpha=1)
		plt.scatter(pick_x, pick_y, s=0.1,c='r')
		plt.title("Simulation XY")
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
		plt.clf()

	def check_mt(data): #D3LDMTのMとTの分布を調べるためのコード(127, 216000, 4)
		m_l = []
		t_l = []
		m = data[:,:,2]
		t = data[:,:,3]
		for i in tqdm(range(m.shape[0])):
			mm = m[i,:]
			tt = t[i,:]
			for j in range(m.shape[1]):
				if mm[j] != 0:
					m_l.append(mm[j])
					t_l.append(tt[j])

		return m_l, t_l

	def plot7featuresLV(data, flist):
		x = data[:,0] #X
		y = data[:,1] #Y
		k = np.log10(data[:,2]) #RHO
		plt.figure(figsize=(16, 4), dpi=300)
		plt.xlim(np.nanmax(x), np.nanmin(x))
		plt.ylim(np.nanmin(y), np.nanmax(y))
		plt.scatter(x, y, s=0.1, alpha=1,c=k,vmin=np.nanmin(k), vmax=np.nanmax(k), cmap='jet')
		plt.colorbar().set_label('I [K]', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		color_list = ["g", "r", "m", '#ff7f00', "#00008B", '#f781bf', "#00CED1"]
		for i in range(7):
			featurex = (flist[i])[:,0]
			featurey = (flist[i])[:,1]
			plt.scatter(featurex, featurey, s=0.1,c = color_list[i])
		plt.title("Simulation (R,θ)=(8.5, -120)")
		plt.xlabel("L [deg.]")
		plt.ylabel("V [km/s]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
		plt.clf()

	def plot7featuresXY(data, flist):
		x = data[:,2] #X
		y = data[:,3] #Y
		plt.figure(figsize=(6, 5), dpi=300)
		plt.xlim(- 9.5, 9.5)
		plt.ylim(- 9.5, 9.5)
		plt.scatter(x, y, s=0.1, alpha=1, c= "k")
		color_list = ["g", "r", "m", '#ff7f00', "#00008B", '#f781bf', "#00CED1"]
		for i in range(7):
			featurex = (flist[i])[:,2]
			featurey = (flist[i])[:,3]
			plt.scatter(featurex, featurey, s=0.1,c = color_list[i])
		plt.title("Simulation XY")
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png',  pad_inches=0) #bbox_inches='tight',
		plt.clf()

	def trans_lvxy(fea_data, lvxy_data, area): #fea_data...[X,Y], areaはkpc
		x_alldata = lvxy_data[:,2]
		y_alldata = lvxy_data[:,3]
		for i in tqdm(range(fea_data.shape[0])):
			seach_x = fea_data[i, 0]
			seach_y = fea_data[i, 1]
			dens_np = np.sqrt(np.square(x_alldata - seach_x) + np.square(y_alldata - seach_y))
			find_index = np.where(dens_np <= area)
			find_l = lvxy_data[find_index, 0]
			find_v = lvxy_data[find_index, 1]
			find_x = lvxy_data[find_index, 2]
			find_y = lvxy_data[find_index, 3]
			find_l = np.reshape(find_l, (find_l.shape[1],1))
			find_v = np.reshape(find_v, (find_v.shape[1],1))
			find_x = np.reshape(find_x, (find_x.shape[1],1))
			find_y = np.reshape(find_y, (find_y.shape[1],1))
			if i == 0:
				find_lv = np.hstack([find_l, find_v])
				find_xy = np.hstack([find_x, find_y])
			if i >= 1:
				find_lv2 = np.hstack([find_l, find_v])
				find_xy2 = np.hstack([find_x, find_y])
				find_lv = np.vstack([find_lv, find_lv2])
				find_xy = np.vstack([find_xy, find_xy2])
		return find_lv, find_xy

	def lv_lv(f_lv, lvk_data):
		for i in tqdm(range(f_lv.shape[0])):
			find_l = f_lv[i,0]
			find_v = f_lv[i,1]
			calculate_np = np.sqrt(np.square(lvk_data[:,0] - find_l) + np.square(lvk_data[:,1] - find_v))
			min_index = np.argmin(calculate_np)
			if i == 0:
				find_lvk = lvk_data[min_index, :]
			if i >= 1:
				find_lvk2 = lvk_data[min_index, :]
				find_lvk = np.vstack([find_lvk, find_lvk2])
		return find_lvk

	def featureXY(data, pick_data, type = 0):
		x = data[:,2] #X
		y = data[:,3] #Y
		pick_x = pick_data[:,0]
		pick_y = pick_data[:,1]
		x_range = np.nanmax(x)
		y_range = np.nanmax(y)
		if type == 0:
			plt.figure(figsize=(6, 5), dpi=300)
		elif type == 1:
			plt.figure(figsize=(6, 5), dpi=100)
		plt.xlim(- x_range - 1, x_range + 1)
		plt.ylim(- y_range - 1, y_range + 1)
		plt.scatter(x, y, s=0.1, alpha=1)
		plt.scatter(pick_x, pick_y, s=1,c='r')
		plt.title("Simulation XY")
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")

		if type == 0:
			dt = datetime.datetime.now()
			image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
			plt.clf()

		elif type == 1:
			plt.show()

	def featureLVK(data, pick_data, type = 0):
		x = data[:,0] #X
		y = data[:,1] #Y
		k = np.log10(data[:,2]) #RHO
		#k = data[:,11] #RHO
		pick_x = pick_data[:,0]
		pick_y = pick_data[:,1]
		if type == 0:
			plt.figure(figsize=(16, 4), dpi=300)
		elif type == 1:
			plt.figure(figsize=(16, 4), dpi=100)
		plt.xlim(np.nanmax(x), np.nanmin(x))
		plt.ylim(np.nanmin(y), np.nanmax(y))
		plt.scatter(x, y, s=0.1, alpha=1,c=k,vmin=np.nanmin(k), vmax=np.nanmax(k), cmap='jet')
		plt.colorbar().set_label('I [K]', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		plt.scatter(pick_x, pick_y, s=0.1,c='red')
		plt.title("Simulation R8θ65")
		plt.xlabel("L [deg.]")
		plt.ylabel("V [km/s]")
		if type == 0:
			dt = datetime.datetime.now()
			image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
			plt.clf()
		elif type == 1:
			plt.show()

	def plot7featuresXYXY(data, flist, title = ""): #data...[LVXY] flist...[XY]
		x = data[:,2] #X
		y = data[:,3] #Y
		plt.figure(figsize=(6, 5), dpi=300)
		plt.xlim(- 9.5, 9.5)
		plt.ylim(- 9.5, 9.5)
		plt.scatter(x, y, s=0.1, alpha=1, c= "k")
		color_list = ["g", "r", "m", '#ff7f00', "#00008B", '#f781bf', "#00CED1"]
		for i in range(7):
			featurex = (flist[i])[:,0]
			featurey = (flist[i])[:,1]
			plt.scatter(featurex, featurey, s=0.1,c = color_list[i])
		plt.title(title)
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', pad_inches=0) #bbox_inches='tight',
		plt.clf()

	def plot7featuresLVLV(data, flist, title =""):
		x = data[:,0] #X
		y = data[:,1] #Y
		k = np.log10(data[:,2]) #RHO
		plt.figure(figsize=(16, 4), dpi=300)
		plt.xlim(np.nanmax(x), np.nanmin(x))
		plt.ylim(np.nanmin(y), np.nanmax(y))
		plt.scatter(x, y, s=0.1, alpha=1,c=k,vmin=np.nanmin(k), vmax=np.nanmax(k), cmap='jet')
		plt.colorbar().set_label('I [K]', labelpad=-38, y=1.04, rotation=0) #labelpadを大きくすると右に。yを大きくすると上に。
		color_list = ["g", "r", "m", '#ff7f00', "#00008B", '#f781bf', "#00CED1"]
		for i in range(7):
			featurex = (flist[i])[:,0]
			featurey = (flist[i])[:,1]
			plt.scatter(featurex, featurey, s=0.1,c = color_list[i])
		plt.title(title)
		plt.xlabel("L [deg.]")
		plt.ylabel("V [km/s]")
		dt = datetime.datetime.now()
		image = plt.savefig(str(dt)+'.png', bbox_inches='tight', pad_inches=0)
		plt.clf()

	def make_dotXY(data, lvk_data, area): #XY平面プロットに点を打つ。データはLVXYを想定
		dot_x, dot_y = [], []
		def onclick(event):
			#print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
			dot_x.append(event.xdata)
			dot_y.append(event.ydata)
			plt.scatter(event.xdata, event.ydata, s=20, c= "r")
			plt.draw()
			print("X: " + str(event.xdata) + ", Y: " + str(event.ydata) + " added!")

		x = data[:,2] #X
		y = data[:,3] #Y
		fig = plt.figure(figsize=(6, 5), dpi=100)
		plt.xlim(np.nanmin(x) - 1, np.nanmax(x) + 1)
		plt.ylim(np.nanmin(y) - 1, np.nanmax(y) + 1)
		plt.scatter(x, y, s=0.1, alpha=1, c= "k")
		plt.title("Simulation XY")
		plt.xlabel("X [kpc]")
		plt.ylabel("Y [kpc]")
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		plt.show()
		d_x = np.array(dot_x)
		d_x = d_x.reshape((d_x.shape[0],1))
		d_y = np.array(dot_y)
		d_y = d_y.reshape((d_y.shape[0],1))
		XY_num = np.concatenate((d_x,d_y),axis=1)
		feature_lv, feature_xy = Basedef.trans_lvxy(XY_num, data, area) #featureデータ(X,Y)に相当するLVXYデータのLVを探す
		match_lvk = Basedef.lv_lv(feature_lv, lvk_data) #LVKデータ上のLVを返す

		Basedef.featureXY(lvxy_data, feature_xy, 1)
		Basedef.featureLVK(lvk_data, match_lvk, 1)

		save_q = input("保存する？ [y/n]\n入力: ")

		if save_q == "y":
			save_title = input("保存名: ")
			np.save(PATH + save_title + ".npy", XY_num)
			print(save_title + ".npy save!")

		else:
			sys.exit()

	def dat_to_npy(message = "datデータを選択"):
		print(message)
		dat_list = list(PWD.glob('*.dat'))
		for i in range(len(dat_list)):
			print(str(i) + ": " + str(dat_list[i]).replace(PATH,""))
		select_number = input("入力: ")
		if not select_number:
			dat = open(dat_list[2], 'r')
			file_name = str(dat_list[2]).replace(PATH,"").replace(".dat","")
		else:
			dat = open(dat_list[int(select_number)], 'r')
			file_name = str(dat_list[int(select_number)]).replace(PATH,"").replace(".dat","")

		i = 0
		for line in tqdm(dat):
			data_list = line.split()
			if i == 0:
				dat_np = np.array(data_list)
			else:
				dat_np2 = np.array(data_list)
				dat_np = np.vstack([dat_np, dat_np2])
			i += 1

		np.save(PATH + file_name + ".npy", dat_np)
		file.close()
		sys.exit()

# class LINENotifyBot:
# 	API_URL = 'https://notify-api.line.me/api/notify'
#     def __init__(self, access_token):
#         self.__headers = {'Authorization': 'Bearer ' + access_token}
#
#     def send(
#             self, message,
#             image=None, sticker_package_id=None, sticker_id=None,
#             ):
#         payload = {
#             'message': message,
#             'stickerPackageId': sticker_package_id,
#             'stickerId': sticker_id,
#             }
#         files = {}
#         if image != None:
#             files = {'imageFile': open(image, 'rb')}
#         r = requests.post(
#             LINENotifyBot.API_URL,
#             headers=self.__headers,
#             data=payload,
#             files=files,
#             )

class Simulator:
	def obsout_pick(data, rad): #観測者のradより内側に存在するデータのみをピックアップ
		r = np.sqrt(np.square(data[:,0]) + np.square(data[:,1]))
		pick_index = np.where(r <= rad)
		new_data = data[pick_index]
		return new_data

	def vel_calculatea(data, radius, area): #areaはradiusから±どれだけの範囲からデータをとるか
		r = np.sqrt(np.square(data[:,0]) + np.square(data[:,1]))
		acc_index = np.where((radius - area <= r) & (r <= radius + area))
		acc_data = data[acc_index]
		acc_x = (i for i in acc_data[:,0])
		acc_y = (j for j in acc_data[:,1])
		acc_ax = (k for k in acc_data[:,8])
		acc_ay = (l for l in acc_data[:,9])
		acc_list = []
		for i, j, k, l in tqdm(zip(acc_x, acc_y, acc_ax, acc_ay)):
			point_vec = np.array([i,j])
			acc_vec = np.array([k,l])
			proje_vecx = np.dot(point_vec, acc_vec) * i / (LA.norm(point_vec) ** 2)
			proje_vecy = np.dot(point_vec, acc_vec) * j / (LA.norm(point_vec) ** 2)
			proje_vec = np.array([proje_vecx, proje_vecy])
			cos = np.dot(proje_vec, point_vec) / (LA.norm(proje_vec) * LA.norm(point_vec))
			if cos == -1: #内向き
				acc_list.append(LA.norm(proje_vec))
			elif cos == 1: #外向き
				acc_list.append(- LA.norm(proje_vec))

		kpc_km = 3.086*(10**16)
		average_a = sum(acc_list) / len(acc_list)
		acc_v = math.sqrt(radius * kpc_km * average_a) #絶対値
		return acc_v

	def make_ld(data, radius, degree, velocity): #VTLDMデータを作るための準備（未グリッド）
		x_i = (i for i in data[:,0]) 	#シミュレーションデータのx座標
		y_i = (j for j in data[:,1])	#y座標
		z_i = (z for z in data[:,2])	#z座標
		vx_i = (k for k in data[:,3])	#V_x
		vy_i = (l for l in data[:,4])	#V_y
		m_i = (m for m in data[:,7])	#M_sun
		t_i = (t for t in data[:,12])	#温度
		x_0 = radius * math.cos(math.radians(degree))	#角度はx軸正方向を0度として反時計回り正, 時計回り負
		y_0 = radius * math.sin(math.radians(degree))	#観測者の位置(X,Y)を角度から出す
		obs_vec = np.array([x_0,y_0])	#観測者ベクトル
		obs_vecm = - obs_vec 	#観測者ベクトル（マイナス）
		obs_degree = math.atan2(y_0, x_0)	#？
		v0_x = velocity * math.sin(obs_degree)	#速度は時計回りが正
		v0_y = - velocity * math.cos(obs_degree)	#観測者の速度（Vx,Vy）
		x_0sign = x_0 / abs(x_0) #観測者のX座標が第1,4象限か2,3か
		degree_list, vel_list, de_list, t_list, m_list = [], [], [], [], []	#VRTLDM

		for i, j, z, k, l, t, m in tqdm(zip(x_i, y_i, z_i, vx_i, vy_i, t_i, m_i)): #4m27s(4625849)17322.46it/s
			vec_a = np.array([i - x_0, j - y_0]) #観測者から各点へのベクトル
			theta_i = np.inner(vec_a, obs_vecm) / (LA.norm(vec_a) * LA.norm(obs_vecm)) #銀経
			vel_vec = np.array([k - v0_x, l - v0_y]) #相対速度ベクトル
			v_lsrvec = np.inner(vel_vec, vec_a) * vec_a / LA.norm(vec_a)**2
			v_lsr = abs(np.inner(vel_vec, vec_a)) / LA.norm(vec_a) #視線速度成分(射影ベクトルの大きさ)
			theta_c = np.inner(v_lsrvec,vec_a) / (LA.norm(v_lsr) * LA.norm(vec_a)) #射影ベクトルと観測者-各点ベクトルのなす角, -1: 観測者向き 1: 同方向
			theta_i2 = x_0sign * np.rad2deg(np.arccos(np.clip(theta_i, -1.0, 1.0))) #象限による符号変更
			theta_b = math.degrees(math.atan2(abs(z), LA.norm(vec_a))) #銀緯1°の範囲でデータを絞るため
			if theta_b > 1:
				continue

			if (y_0 * i / x_0) <= j: #上側
				theta_i2 = - theta_i2 #観測者に対する位置の符号変更

			degree_list.append(theta_i2)
			vel_list.append(theta_c * v_lsr)
			de_list.append(LA.norm(vec_a))
			t_list.append(t)
			m_list.append(m) #M_sun

		d_l = Basedef.list_transpose(degree_list)
		v_l = Basedef.list_transpose(vel_list)
		de_l = Basedef.list_transpose(de_list)
		t_l = Basedef.list_transpose(t_list)
		m_l = Basedef.list_transpose(m_list)
		VTLDM_simu = np.concatenate((v_l, t_l, d_l, de_l, m_l),axis=1)
		dt_now = datetime.datetime.now()
		vtldm_name = "VTLDM("+ str(dt_now) + ").npy"
		np.save(vtldm_name, VTLDM_simu)
		return vtldm_name

	def sort_data(data, v_step): #VTLDMデータをVについて順番に並べてv_step(1.3km/s)で区切り平均をとる
		v_sortdata = data[np.argsort(data[:, 0])]
		v_start = v_sortdata[0,0] #-111.726570668
		v_end = v_sortdata[-1,0] #225.050031837
		v_list = np.arange(v_start, v_end + v_step, v_step) #-111.72657067-225.87342933, shape:(212,)
		new_v, new_other = [], []

		for i in tqdm(range(v_list.size - 1)):
			v_st = v_list[i]
			v_en = v_list[i + 1]
			v_mid = ( v_st + v_en ) / 2
			new_v.append(v_mid)
			vselect_index = np.where((v_st <= v_sortdata[:,0]) & (v_sortdata[:,0] <= v_en))
			v_select_data = v_sortdata[vselect_index]
			other_l = []

			for j in range(v_select_data.shape[0] - 1): #TLDM, 備考(Rはmax: 154281.2, min: 1.417125e-12)
				ap_list = [v_select_data[j,1], v_select_data[j,2], v_select_data[j,3], v_select_data[j,4]]
				other_l.append(ap_list)

			new_other.append(other_l)

		N_V = Basedef.list_transpose(new_v)
		N_O = Basedef.list_transpose(new_other)
		v_arrangedata = np.concatenate((N_V, N_O), axis=1)
		dt_now = datetime.datetime.now()
		sort_name = "Sort_VTLDM("+ str(dt_now) + ").npy"
		np.save(sort_name, v_arrangedata)
		return sort_name

	def make_vlrdt(data, l_st, l_en): #[V,[T,L,D,M]]を[V,[L,D,T,M]](各Vについてグリッド分けした全範囲のLDを+)に変える
		def make_vbase():
			l_grid = 0.125 #元は0.02361111111, 短縮のため0.125に
			d_grid = 0.1 #100pcと仮定→2019/11/30, 50pcに
			d_start = 0
			d_end = 30
			l_mesh = np.arange(l_st, l_en + l_grid, l_grid) #[ 10. 10.02361111 10.04722222 ...,  49.97361111  49.99722222 50.02083333] shape(1696,)
			d_mesh = np.arange(d_end, d_start - d_grid, - d_grid) #[30 ,..., 1.00000000e-01  -4.26325641e-13] shape(301,)
			grid_l, grid_d, grid_n, grid_t = [], [], [], []

			for i in tqdm(range(l_mesh.shape[0] - 1)): #-1しないとエラー
				l_start = l_mesh[i]
				l_end = l_mesh[i + 1]
				l_mid = (l_start + l_end) / 2

				for j in range(d_mesh.shape[0] - 1):
					d_start = d_mesh[j]
					d_end = d_mesh[j+1]
					d_mid = (d_start + d_end) / 2

					grid_l.append(l_mid)
					grid_d.append(d_mid)
					grid_n.append(0)
					grid_t.append(0)

			l_l = Basedef.list_transpose(grid_l)
			d_l = Basedef.list_transpose(grid_d)
			n_l = Basedef.list_transpose(grid_n)
			t_l = Basedef.list_transpose(grid_t)
			v_base = np.concatenate((l_l,d_l,n_l,t_l),axis=1) #shape: (508500, 4)
			return v_base

		v_list, LD_list = [], [] #VのリストとLDグリッドは分けて出力する
		v_base = make_vbase()
		l_grid = 0.125 #元は0.02361111111, 短縮のため0.125に
		d_grid = 0.1 #100pcと仮定

		for i in tqdm(range(data.shape[0] - 1)): #ここでのデータはVについて区切った[[V,[T,L,D,M]],...]
			v_list.append(data[i,0]) #-110.92...から開始(次: -109.326...)
			other_data = np.array(data[i,1]) #other_dataは[[TLDM],...]
			ap_vbase = copy.deepcopy(v_base) #LDNT(LDMT)配列

			for j in range(other_data.shape[0] - 1): #ap_vbaseはLDRT(※: LDMT)の順

				search_l = other_data[j, 1] #np arrayの[[TLDM],...]
				l_match = np.where((ap_vbase[:, 0] <= search_l) & (search_l <= ap_vbase[:, 0] + l_grid)) #Lの範囲で探す
				change_data = ap_vbase[l_match] #ap_vbaseのLの範囲で捕まったデータ（複数ある場合がある）RTLDのLで捕捉

				if change_data.size == 0:
					continue

				search_d = other_data[j, 2] #T,L,D,Mの順
				d_match = np.where((change_data[:,1] - d_grid <= search_d) & (search_d <= change_data[:,1])) #Dの範囲で探す
				change_data2 = change_data[d_match]

				if change_data2.size == 0:
					continue


				change_index = np.where((ap_vbase[:,0] == change_data2[0, 0]) & (ap_vbase[:,1] == change_data2[0, 1]))
				if ap_vbase[change_index, 2] != 0: #ap_vbaseはLDNT(LDMT)配列
					old_r = ap_vbase[change_index, 2] #rじゃなくて今はm
					old_t = ap_vbase[change_index, 3]
					ap_vbase[change_index, :] = [change_data2[0, 0], change_data2[0, 1], other_data[j, 3] + old_r, (other_data[j, 0] + old_t) / 2]
					continue
				ap_vbase[change_index, :] = [change_data2[0, 0], change_data2[0, 1], other_data[j,3], other_data[j,0]]
			LD_list.append(ap_vbase)

		LD_numpy = np.array(LD_list)
		v_numpy = np.array(v_list)

		dt_now = datetime.datetime.now()
		d3_name = "D3_LDMT("+ str(dt_now) + ").npy"
		v_name = "D1_V("+ str(dt_now) + ").npy"
		np.save(d3_name, LD_numpy)
		np.save(v_name, v_numpy)
		return d3_name, v_name

	def add_pole(data): #LDMTNを作ろうとしたらメモリエラー。Vと同じようにNだけ独立に作る。D3_LDMTの形: [210, 508500, 4]
		slice_data = data[:,:,2] #LDMTのM面
		pole_l = []
		grid_l = 0.125 #元は0.02361111111, 短縮のため0.125に
		for i in tqdm(range(slice_data.shape[0])):
			pole_ll = []
			for j in range(slice_data.shape[1]):
				change_m = slice_data[i,j]
				slice_d = data[i, j, 1]
				range_a = slice_d * math.sin(math.radians(grid_l)) #pythonのmath.sinの中身はラジアン
				range_b = slice_d * math.tan(math.radians(1))
				volume_kpc3 = range_a * range_b * 0.1 #0.1kpc→0.05kpcに変更（2019/11/30）
				volume_cm3 = volume_kpc3 * ((3.0 * (10 ** 21)) ** 3)
				volume_density = (change_m * (1.989 * (10 ** 30))) / volume_cm3
				h_density = volume_density / (1.674 * 2 * (10 ** (- 27)))
				pole_density = h_density * (0.1 * 3.0 * (10 ** 21)) * (8 * (10 ** (- 5)))
				pole_ll.append(pole_density)
			pole_l.append(pole_ll)
		pole_np = np.array(pole_l)
		pole_np = np.reshape(pole_np, (pole_np.shape[0], pole_np.shape[1], 1))
		dt_now = datetime.datetime.now()
		pole_name = "D1_N("+ str(dt_now) + ").npy"
		np.save(pole_name, pole_np)
		return pole_name

	def add_tau(data, d3_name): #光学的厚み...求めるのにTとNのデータが必要
		load_data = Basedef.npy_search(d3_name) #温度データを含むD3_LDMT([210, 508500, 4])を読み込む
		temp_data = load_data[:,:,3] #shape: (210, 508500)
		slice_data = data[:,:,0] #3次元(210, 508500, 1)のNデータを2次元に, shape: (210, 508500)
		tau_list = []
		for i in tqdm(range(temp_data.shape[0])):
			small_list = []
			for j in range(temp_data.shape[1]):
				temp_num = temp_data[i, j]
				pole_num = slice_data[i, j]
				const_num = (5.53 * 4 * (math.pi ** 3) * ((0.122 * 10 ** (-18)) ** 2) * 2) / (3 * (6.62 * 10 ** (-34)))
				exp_num = (1 - math.exp(- (2 * 5.53) / temp_num)) / math.exp( 5.53 / temp_num)
				tau_num = (const_num * exp_num * pole_num) / (temp_num * 1.3 * (10**(12)))
				if tau_num != tau_num:
					small_list.append(0)
					continue
				small_list.append(tau_num)
				#print(tau_num)
			tau_list.append(small_list)
		tau_npy = np.array(tau_list)
		dt_now = datetime.datetime.now()
		tau_name = "D1_tau("+ str(dt_now) + ").npy"
		np.save(tau_name, tau_npy)
		return tau_name

	def make_LDTtau(data, tau_name):
		tau_data = Basedef.npy_search(tau_name) #D1_tauを読み込む
		ldeg_data = data[:,:,0]
		dist_data = data[:,:,1]
		temp_data = data[:,:,3]
		LDTtau_npy = np.stack([ldeg_data, dist_data, temp_data, tau_data])
		print(LDTtau_npy.shape)
		dt_now = datetime.datetime.now()
		ldttau_name = "D3_LDTtau("+ str(dt_now) + ").npy"
		np.save(ldttau_name, LDTtau_npy)
		return ldttau_name

	def make_LVI(data, v_name, obs_rad, obs_deg, l_st, l_en): #dataはLDTtau
		vel_data = Basedef.npy_search(v_name) #D1_Vを読み込む shape:(210,)
		#vel_data = Basedef.npy_select("Vのデータを選択")

		lde_data = data[0,:,:] #shape: (210, 508500)
		dis_data = data[1,:,:]
		tem_data = data[2,:,:] #T
		tau_data = data[3,:,:] #tau
		t_bg = 2.7
		integ_list = []
		big_orig_box = []
		for i in tqdm(range(dis_data.shape[0])): #(210, 508500)LとD...１つのLについて0-30のDがある。それが10-50のL分ある。
			small_integ = []
			int_harfway = 0
			tau_check_d = [] #tauの値をチェックするためのリスト
			tau_check_tau = [] #tauの値をチェックするためのリスト
			for j in range(dis_data.shape[1] - 1): #(210, 508500)

				i_temp = tem_data[i,j]
				i_tau = tau_data[i,j]

				if i_tau > 0: #LとDから相当するXYを求める
					orig_box = []
					x, y = Simulator.ldxy_cal(lde_data[i, j], dis_data[i, j], obs_rad, obs_deg)
					#print("L: " + str(lde_data[i, j]) + ", D: " + str(dis_data[i, j]) + "⇒ X: " + str(x) + ", Y: " +str(y))
					l = lde_data[i,j]
					v = vel_data[i]
					orig_box.append(l)
					orig_box.append(v)
					orig_box.append(x)
					orig_box.append(y)
					big_orig_box.append(orig_box)

					integ = int_harfway * math.exp(- i_tau) + i_temp * (1 - math.exp(- i_tau))
					int_harfway = integ
					#一つのintegの値がどれだけのτの値からできているか調べる...
					tau_check_d.append(dis_data[i, j])
					tau_check_tau.append(i_tau)


				if dis_data[i,j] <= dis_data[i,j+1]: #Dが0に到達したパターン
					integ = int_harfway * math.exp(- i_tau) + i_temp * (1 - math.exp(- i_tau))
					small_integ.append(integ)

					if len(tau_check_d) >= 10:
						tau_check_d = Basedef.list_transpose(tau_check_d)
						tau_check_tau = Basedef.list_transpose(tau_check_tau)
						check_d_np = np.array(tau_check_d)
						check_tau_np = np.array(tau_check_tau)
						plt.xlim(np.nanmin(check_d_np) - 1, np.nanmax(check_d_np) + 1)
						plt.ylim(np.nanmin(check_tau_np) - 1, np.nanmax(check_tau_np) + 1)
						plt.figure(figsize=(6, 5), dpi=300)
						plt.scatter(check_d_np, check_tau_np, s=10, c="k")
						plt.title("check_tau D-tau (L,V)=(%s deg., %s km/s) in (r,θ)=(%s kpc, %s deg.)" %(int(lde_data[i,j]), int(vel_data[i]), obs_rad, obs_deg))
						plt.xlabel("D [kpc.]")
						plt.ylabel("tau")
						dt = datetime.datetime.now()
						image = plt.savefig("tau_check_%s.png" %(str(dt)), bbox_inches='tight', pad_inches=0)
						plt.clf()
					tau_check_d = []
					tau_check_tau = []

				if j == 0 or (dis_data[i,j] >= dis_data[i,j-1]):
					integ = t_bg * math.exp(- i_tau) + i_temp * (1 - math.exp(- i_tau))
					int_harfway = integ

				else:
					integ = int_harfway * math.exp(- i_tau) + i_temp * (1 - math.exp(- i_tau))
					int_harfway = integ

			integ_list.append(small_integ)
		orig_np = np.array(big_orig_box)
		integ_npy = np.array(integ_list) #shape: (210, 508500)
		np.save("orig_np_"+str(l_st)+"~"+str(l_en)+".npy", orig_np)
		np.save("integ_npy.npy", integ_npy)
		print("integ_npy SAVE!")
		integ_npy = np.load("integ_npy.npy", allow_pickle = True)
		print("integ_npy LOAD!")
		#LVIを作る
		l_list, v_list, i_list = [], [], []
		for i in tqdm(range(vel_data.shape[0])):
			vel_num = vel_data[i]
			m = 0
			for j in range(lde_data.shape[1] - 1):
				if lde_data[i,j] != lde_data[i,j+1]:
					l_list.append(lde_data[i,j])
					v_list.append(vel_num)
					i_list.append(integ_npy[i,m])
					m += 1
				else:
					continue

		l_l = Basedef.list_transpose(l_list)
		v_l = Basedef.list_transpose(v_list)
		i_l = Basedef.list_transpose(i_list)
		print(l_l.shape)
		print(v_l.shape)
		print(i_l.shape)
		new_LVI = np.concatenate((l_l,v_l,i_l),axis=1)
		dt_now = datetime.datetime.now()
		lvi_name = "NEW_LVI("+ str(dt_now) + ").npy"
		np.save(lvi_name, new_LVI)
		return lvi_name

	def ldxy_cal(lon, dis, obs_rad, obs_deg):
		obs_x = obs_rad * math.cos(math.radians(obs_deg))
		obs_y = obs_rad * math.sin(math.radians(obs_deg))

		if obs_deg >= 0 and obs_deg <= 90: #第一象限
			lon_dash = 90 - lon - obs_deg
			target_x = dis * math.sin(math.radians(lon_dash))
			target_y = dis * math.cos(math.radians(lon_dash))
			orig_x = obs_x - target_x
			orig_y = obs_y - target_y

		elif obs_deg >= 90 and obs_deg <= 180:
			lon_dash = lon + obs_deg +-90
			target_x = dis * math.sin(math.radians(lon_dash))
			target_y = dis * math.cos(math.radians(lon_dash))
			orig_x = obs_x + target_x
			orig_y = obs_y - target_y

		else: #第3,4象限ワカンネ
			lon_dash = lon + obs_deg +-90
			target_x = dis * math.sin(math.radians(lon_dash))
			target_y = dis * math.cos(math.radians(lon_dash))
			orig_x = obs_x + target_x
			orig_y = obs_y - target_y

		return orig_x, orig_y

if __name__ == "__main__":

	what_do = input("[1]: プロットしたり色々\n[2]: featureを色々\n入力: ")

	if what_do == "2": #7つのfeatureファイルからLVXYプロットを行う
		what_do2 = input("[1]: 1つのfeatureでプロット\n[2]: 7つのfeatureでプロット\n[3]: XY座標のfeatureをLV図にプロット\n[4]: XY座標の7つのfeatureで\n[5]: feature抽出\n入力: ")
		if what_do2 == "1":
			#featureが一つのパターン
			xy_data = Basedef.npy_select("LVXYデータを選択")
			lv_data = Basedef.npy_select("LVIデータを選択")
			feature_data = Basedef.npy_select("feature抽出データを選択")
			#抽出点の周辺の点に対応する点(X,Y)をLVXYから読み取り
			match_lvxy = Basedef.match_dotsearch(feature_data, xy_data) #LVXYがreturnされる
			#XYに重ねてプロット
			Basedef.double_plotXY(xy_data, match_lvxy)
			#LVに抽出点を重ねてプロット
			Basedef.double_plot(lv_data, match_lvxy)
			sys.exit()

		elif what_do2 == "2":
			f_list = []
			xy_data = Basedef.npy_select("LVXYデータを選択")
			lv_data = Basedef.npy_select("LVIデータを選択")
			for i in range(7):
				#featureの点を抽出したnpy, LVXY, LVIを読み込む
				feature_data = Basedef.npy_select("feature抽出データを選択")
				#抽出点の周辺の点に対応する点(X,Y)をLVXYから読み取り
				match_lvxy = Basedef.match_dotsearch(feature_data, xy_data) #LVXYがreturnされる
				f_list.append(match_lvxy)

			Basedef.plot7featuresLV(lv_data, f_list)
			Basedef.plot7featuresXY(xy_data, f_list)
			sys.exit()

		elif what_do2 == "3":
			xy_feature = Basedef.npy_select("XY座標のfeatureデータを選択")
			lvxy_data = Basedef.npy_select("LVXYデータを選択")
			lvk_data = Basedef.npy_select("LVKデータを選択")
			feature_lv, feature_xy = Basedef.trans_lvxy(xy_feature, lvxy_data, 0.5) #featureデータ(X,Y)に相当するLVXYデータのLVを探す
			match_lvk = Basedef.lv_lv(feature_lv, lvk_data) #LVKデータ上のLVを返す
			Basedef.featureXY(lvxy_data, feature_xy)
			Basedef.featureLVK(lvk_data, match_lvk)
			sys.exit()

		elif what_do2 == "4":
			fxy_list = []
			flvk_list = []
			lvxy_data = Basedef.npy_select("LVXYデータを選択")
			lvk_data = Basedef.npy_select("LVKデータを選択")
			for i in range(7):
				xy_feature = Basedef.npy_select("XY座標のfeatureデータを選択")
				feature_lv, feature_xy = Basedef.trans_lvxy(xy_feature, lvxy_data, 0.5) #featureデータ(X,Y)に相当するLVXYデータのLVを探す
				match_lvk = Basedef.lv_lv(feature_lv, lvk_data) #LVKデータ上のLVを返す
				fxy_list.append(feature_xy)
				flvk_list.append(match_lvk)
			Basedef.plot7featuresXYXY(lvxy_data, fxy_list)
			Basedef.plot7featuresLVLV(lvk_data, flvk_list)
			sys.exit()

		elif what_do2 == "5":
			lvxy_data = Basedef.npy_select("LVXYデータを選択")
			lvk_data = Basedef.npy_select("LVKデータを選択")
			Basedef.make_dotXY(lvxy_data, lvk_data, 0.25)


	elif what_do == "1":

		print("データを選択: ")
		simu_data = Basedef.npy_select()

		rad_list = [-120]
		# rad_list = [-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
		for i in rad_list: #13
		#for i in range(rad_list): #13

			obs_rad = 8.5 #観測者の原点からの位置
			obs_are = 0.1 #速度情報の値を参照するエリア[kpc]
			obs_deg = i
			l_st = - 90
			l_en = 90

			os.chdir(PATH)
			new_folder = "R%sθ%s (%s)" %(obs_rad, obs_deg, datetime.datetime.now())
			new_PATH = PATH + new_folder
			os.mkdir(new_PATH)
			os.chdir(new_PATH)

			#bot = LINENotifyBot(access_token = ACCESS_TOKEN)

			simu_data1 = Simulator.obsout_pick(simu_data, obs_rad)

			#select_index = np.where((simu_data1[:,12] <= 100) & (simu_data1[:,11] >= 10)) #制限変更（2019/11/27）
			select_index = np.where((simu_data1[:,12] <= 100) & (simu_data1[:,11] >= 50)) #FUGINのC18Oと比較するため
			simu_data1 = simu_data1[select_index]
			Basedef.obspos_plot(simu_data1, obs_rad, obs_deg)

			obs_vel = Simulator.vel_calculatea(simu_data1, obs_rad, obs_are)
			VTLDM_name = Simulator.make_ld(simu_data1, obs_rad, obs_deg, obs_vel)

			#領域に分けてループならここから（例: lst=-90 l_en=0）
			cutted_name = Basedef.npy_cut(Basedef.npy_search(VTLDM_name), l_st, l_en)
			#bot.send(message='1/7: new_VTLDM Done.')

			#VTLDMデータをVについて順番に並べて1.3km/sで区切り平均をとる
			sort_name = Simulator.sort_data(Basedef.npy_search(cutted_name), 1.3)
			#bot.send(message='2/7: Sort_VTLDM Done.')

			#各VごとにLDグリッドを用意し、値をもつ場所を探しその値を更新するmake_VLDRT
			d3_name, v_name = Simulator.make_vlrdt(Basedef.npy_search(sort_name), l_st, l_en)
			#bot.send(message='3/7: D3_LDRT Done.')

			#LDの3次元データ[210, 508500, 4]から柱密度Nのデータをつくる
			poledens_name = Simulator.add_pole(Basedef.npy_search(d3_name)) #(210, 508500, 1) Basedef.npy_search(d3_name)
			#bot.send(message='4/7:D 1_N Done.')

			#Nのデータ(210, 508500, 1)から光学的厚みを求める。積分強度を求めることを考えるとLDTtau（1つのファイル）を作っておきたいところ。
			tau_name = Simulator.add_tau(Basedef.npy_search(poledens_name), d3_name) #Basedef.npy_search(poledens_name), d3_name
			#bot.send(message='5/7: D1_tau Done.')

			#LDRTとtauからIを求めるのに必要なLDTtauを作る
			LDTtau_name = Simulator.make_LDTtau(Basedef.npy_search(d3_name), tau_name) #0: LDRT選択 コード内: tau選択
			#bot.send(message='6/7: D3_LDTtau Done.')

			#複写輸送方程式を解いてIを求め、LVIデータを作る
			LVI_name = Simulator.make_LVI(Basedef.npy_search(LDTtau_name), v_name, obs_rad, obs_deg, l_st, l_en) #選択: D3_LDTtau
			u2cut_name = Basedef.u2k_cut(LVI_name)
			#bot.send(message='7/7: NEW_LVI & U2cut Done!')

			#plot_image = Basedef.plot(Basedef.npy_search(LVI_name), obs_rad, obs_deg) #Basedef.plotの変数受け渡しは要変更
			plot_image2 = Basedef.plot(Basedef.npy_search(u2cut_name), obs_rad, obs_deg)
			os.remove(new_PATH + "/" + d3_name)
			os.remove(new_PATH + "/" + LDTtau_name)

			#bot.send(message='Mission Complete!',image= "0-10.png", sticker_package_id=1, sticker_id=13,) #画像をnpy_searchみたいに探してプロット結果を送りたい
			#bot.send(message="All Done!!")
		sys.exit()

	else:
		sys.exit()