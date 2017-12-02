use std::fs;
use std::io;
use std::path;

extern crate byteorder;
extern crate flate2;

use self::byteorder::ByteOrder;
use std::io::Read;

const SIZE: usize = 28;

pub struct Dataset {
    images: Vec<[u8; SIZE * SIZE]>,
    labels: Vec<u8>,
}

pub struct MNIST {
    pub train: Dataset,
    pub test: Dataset,
}

fn parse_images<P>(path: P) -> io::Result<Vec<[u8; SIZE * SIZE]>>
    where P: AsRef<path::Path>
{
    let mut decoder = flate2::read::GzDecoder::new(fs::File::open(path)?);
    {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        let magic = byteorder::BigEndian::read_u32(&buf);
        if magic != 2051 {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    let n_images = {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        byteorder::BigEndian::read_u32(&buf) as usize
    };
    {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        let height = byteorder::BigEndian::read_u32(&buf) as usize;
        if height != SIZE {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        let width = byteorder::BigEndian::read_u32(&buf) as usize;
        if width != SIZE {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    let mut images = Vec::with_capacity(n_images);
    for _ in 0..n_images {
        let mut buf = [0; SIZE * SIZE];
        decoder.read_exact(&mut buf)?;
        images.push(buf);
    }
    Ok(images)
}

fn parse_labels<P>(path: P) -> io::Result<Vec<u8>>
    where P: AsRef<path::Path>
{
    let mut decoder = flate2::read::GzDecoder::new(fs::File::open(path)?);
    {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        let magic = byteorder::BigEndian::read_u32(&buf);
        if magic != 2049 {
            return Err(io::ErrorKind::InvalidData.into());
        }
    }
    let n_items = {
        let mut buf = [0; 4];
        decoder.read_exact(&mut buf)?;
        byteorder::BigEndian::read_u32(&buf) as usize
    };
    let mut labels = vec![0; n_items];
    decoder.read_exact(&mut labels)?;
    Ok(labels)
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.images.len()
    }

    pub fn get(&self, index: usize) -> (&[u8; SIZE * SIZE], u8) {
        (&self.images[index], self.labels[index])
    }
}

impl MNIST {
    pub fn new<P>(path: P) -> io::Result<MNIST>
        where P: AsRef<path::Path>
    {
        let path = path.as_ref();

        let train = {
            let images = parse_images(path.join("train-images-idx3-ubyte.gz"))?;
            let labels = parse_labels(path.join("train-labels-idx1-ubyte.gz"))?;
            if images.len() != labels.len() {
                return Err(io::ErrorKind::InvalidData.into());
            }
            Dataset { images, labels }
        };

        let test = {
            let images = parse_images(path.join("t10k-images-idx3-ubyte.gz"))?;
            let labels = parse_labels(path.join("t10k-labels-idx1-ubyte.gz"))?;
            if images.len() != labels.len() {
                return Err(io::ErrorKind::InvalidData.into());
            }
            Dataset { images, labels }
        };

        Ok(MNIST { train, test })
    }

    pub const SIZE: usize = SIZE;
}
