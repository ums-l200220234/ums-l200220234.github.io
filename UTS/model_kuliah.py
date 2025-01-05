from metaflow import FlowSpec, step

class KuliahFlow(FlowSpec):
    @step
    def start(self):
        print("Terdaftar sebagai Mahasiswa, lakukan pembayaran SPP untuk melanjutkan ke perkuliahan")
        self.pembayaran_spp = True  
        self.next(self.konfirmasi_pembayaran)
    
    @step
    def konfirmasi_pembayaran(self):
        if self.pembayaran_spp:
            print("Pembayaran SPP telah dikonfirmasi")
            print("Lanjutkan ke perkuliahan...")
        else:
            print("Pembayaran SPP belum dikonfirmasi")
            print("Kuliah tidak dapat dilanjutkan")
            self.next(self.end)
        self.next(self.perkuliahan)
    
    @step
    def perkuliahan(self):
        self.kehadiran = 0
        self.hadir = True
        self.total_pertemuan = 14

        while self.kehadiran < self.total_pertemuan:
            if self.hadir:
                self.kehadiran += 1
        if self.kehadiran < 14 * 0.75:
            self.next(self.end)
            return
        self.next(self.tugas)
    
    @step
    def tugas(self):
        self.nilai_tugas = 85
        print(f"Nilai tugas: {self.nilai_tugas}")
        self.next(self.uts)
        

    @step
    def uts(self):
        self.nilai_uts = 75 
        print(f"Nilai UTS: {self.nilai_uts}")
        self.next(self.uas)

    @step
    def uas(self):
        self.nilai_uas = 80
        print(f"Nilai UAS: {self.nilai_uas}")
        self.next(self.menghitung_nilai_akhir)

    @step
    def menghitung_nilai_akhir(self):
        persen_kehadiran = self.kehadiran / self.total_pertemuan * 100
        nilai_presensi = (persen_kehadiran / 100) * 10  
        
        self.nilai_akhir = (0.2 * self.nilai_tugas) + (0.3 * self.nilai_uts) + (0.5 * self.nilai_uas) + nilai_presensi
        
        print(f"Nilai akhir: {self.nilai_akhir} (Termasuk nilai presensi: {nilai_presensi})")
        
        # Tentukan huruf nilai
        if self.nilai_akhir >= 80:
            self.huruf_nilai = 'A'
        elif self.nilai_akhir >= 70:
            self.huruf_nilai = 'B'
        elif self.nilai_akhir >= 60:
            self.huruf_nilai = 'C'
        elif self.nilai_akhir >= 50:
            self.huruf_nilai = 'D'
        else:
            self.huruf_nilai = 'F'
        
        print(f"Huruf nilai: {self.huruf_nilai}")
        
        self.next(self.end)
    
    
    @step
    def end(self):
        if self.nilai_akhir < 60: 
            print(f"Nilai akhir anda adalah {self.huruf_nilai}, silakan ulang proses kuliah.")
            self.next(self.start)
        else:
            print("Proses kuliah selesai.")

if __name__ == '__main__':
    KuliahFlow()
