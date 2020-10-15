#ifndef INCLUDED_TOOLERROR_H
#define INCLUDED_TOOLERROR_H

/** common error codes */
#define TmToolSuccess                    0x00000000
#define TmToolError_Failure              0x00000001
#define TmToolError_Timeout              0x00000002
#define TmToolError_DeviceNotFound       0x00000003
#define TmToolError_MallocFail           0x00000004
#define TmToolError_InitLogFail          0x00000005
#define TmToolError_IoctlFailed          0x00000006
#define TmToolError_FileWriteFailed      0x00000007
#define TmToolError_FileReadFailed       0x00000008
#define TmToolError_InvalidAddress       0x00000009

#endif // INCLUDED_DLAERROR_H