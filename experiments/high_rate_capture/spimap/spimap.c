// SPDX-License-Identifier: GPL-2.0
/*
 * Exposes the registers of the MediaTek SPI controller 0 (0x1100c000, 4 KB) to user space through mmap, for the
 * Portiloop capture experiment. The kernel spi-mt65xx driver stays bound: the user program must keep the
 * controller's interrupts disabled while it drives the controller, and the controller must stay powered
 * (power/control = on).
 */
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>

#define SPI0_PHYS 0x1100c000UL
#define SPI0_SIZE 0x1000UL

static int spimap_mmap(struct file *file, struct vm_area_struct *vma)
{
	if (vma->vm_pgoff != 0 || vma->vm_end - vma->vm_start != SPI0_SIZE)
		return -EINVAL;
	vma->vm_page_prot = pgprot_device(vma->vm_page_prot);
	return io_remap_pfn_range(vma, vma->vm_start, SPI0_PHYS >> PAGE_SHIFT, SPI0_SIZE, vma->vm_page_prot);
}

static const struct file_operations spimap_fops = {
	.owner = THIS_MODULE,
	.mmap = spimap_mmap,
};

static struct miscdevice spimap_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "spimap",
	.fops = &spimap_fops,
	.mode = 0600,
};

static int __init spimap_init(void)
{
	return misc_register(&spimap_dev);
}
module_init(spimap_init);

MODULE_DESCRIPTION("mmap of the MediaTek SPI controller 0 registers (Portiloop capture experiment)");
MODULE_LICENSE("GPL");
